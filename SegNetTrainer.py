import os
import numpy as np
import time

from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SegDatasetGenerator import DatasetGenerator
import matplotlib
from dice_loss import dice_coeff
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------- 

class SegNetTrainer():
    def __init__(self):
        self.lossMIN=100000
        self.resume = 0

    def set_arch(self, arch, classes, init, pretrained=False, verbose=True):
        if init == 'transfer':
            transfer = True
        else:
            transfer = None
        if arch == 'deeplabv3_resnet101':
            model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=classes, pretrained=pretrained)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        device = torch.device("cuda:0")
        model.to(device)
        if verbose == True:
            print('Using ' + arch + ' architecture')
        return model

    def train(train_fn, val_fn, nn_arch, nn_pretrain, nn_classes, tr_batchsize, tr_epochs, resize_shape, crop_shape, modelfn, init, pathCkpt,nameCkpt,transfer):
        if nn_arch=='deeplabv3_resnet101':
            model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=nn_classes, pretrained=nn_pretrain)
        model = torch.nn.DataParallel(model,device_ids=[0,1,2])
        device = torch.device("cuda:0")
        model.to(device)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathDatasetFile=train_fn, augment=True)
        datasetVal = DatasetGenerator(pathDatasetFile=val_fn, augment=False)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=tr_batchsize, shuffle=True,  num_workers=1, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=tr_batchsize, shuffle=False, num_workers=1, pin_memory=True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.000005, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        optimizer = optim.RMSprop(model.parameters(), lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = nn.BCEWithLogitsLoss()

        # ------------------- INITIALIZATION
        resume=0
        if init == 'checkpoint':
            resume = int(os.path.basename(modelfn).split(nameCkpt)[1].split('.')[0])
            print('resuming from ' + modelfn + ' epoch ' + str(resume))
            modelCheckpoint = torch.load(modelfn)
            model.load_state_dict(modelCheckpoint['state_dict'], strict=False)
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
        elif init =='warmstart':
            print('warmstarting from ' + modelfn)
            modelCheckpoint = torch.load(modelfn)
            model.load_state_dict(modelCheckpoint['state_dict'], strict=False)

        # Print Parameters
        print('Training size: ' + str(datasetTrain.__len__()))
        print('Validation size: ' + str(datasetVal.__len__()))
        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (resume, tr_epochs):

            SegNetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, tr_epochs, nn_classes, loss)
            lossVal = SegNetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, tr_epochs, nn_classes, loss)
            
            scheduler.step(lossVal)

            # Save if lowest loss
            if lossVal < lossMIN:
                lossMIN = lossVal    
                savename = os.path.join(pathCkpt,nameCkpt + str(epochID + 1).zfill(2) +'.pth' )
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, savename)#'m-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] loss= ' + str(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + '] loss= ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        for batchID, (img, seg, imgfn, segfn) in enumerate (dataLoader):
            # print(str(batchID) + '/' + str(dataLoader.__len__()))

            seg_probs_cuda = model.forward(img.cuda())
            seg_probs = seg_probs_cuda['out'].cpu()

            seg_true = torch.from_numpy(to_categorical(np.squeeze(seg, axis=1), num_classes=2))

            lossvalue = loss(seg_probs, seg_true)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval()
        with torch.no_grad():
            outLoss = 0

            for i, (img, seg, imgfn, segfn) in enumerate (dataLoader):
                seg_probs_cuda = model.forward(img.cuda())
                seg_probs = seg_probs_cuda['out'].cpu()
                seg_true = torch.from_numpy(to_categorical(np.squeeze(seg, axis=1), num_classes=2))
                lossvalue = loss(seg_probs, seg_true)
                optimizer.zero_grad()
                outLoss += lossvalue.item()
        return outLoss
    
    #--------------------------------------------------------------------------------  

    def test(tst_fn, pathModel, nn_arch, nn_classes,  trBatchSize, transResize, transCrop, outcsv):

        cudnn.benchmark = True
        device = torch.device("cuda:0")

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nn_arch=='deeplabv3_resnet101':
            model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=nn_classes, pretrained=False)

        model = torch.nn.DataParallel(model,device_ids=[0,1,2])
        model.to(device)

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        datasetTest = DatasetGenerator(pathDatasetFile=tst_fn, augment=False)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=0, shuffle=False, pin_memory=True)

        # -------------------- SETTINGS: LOSS
        loss = nn.BCEWithLogitsLoss()

        model.eval()
        outLoss = 0
        dice=0
        num=0
        for i, (img, seg, imgfn, lblfn) in enumerate(dataLoaderTest):

            seg_probs_cuda = model.forward(img.cuda())
            seg_probs = seg_probs_cuda['out'].cpu()

            seg_true = torch.from_numpy(to_categorical(np.squeeze(seg, axis=1), num_classes=2))

            lossvalue = loss(seg_probs, seg_true)
            outLoss += lossvalue.item()

            dice += dice_coeff(np.argmax(seg_probs.cpu().detach(), 1), seg_true[:,1])
            num += len(imgfn)

            # Save input img, input mask, output mask
            for n in range(0,len(imgfn)):
                # Save input img

                imgshow = img[n].permute(1, 2, 0).detach().numpy()
                imgshow = imgshow-imgshow.min()
                imgshow = imgshow/imgshow.max()

                maskshow = np.asarray(seg[n])

                maskpredshow = seg_probs[n].detach().numpy()
                class_pred = maskpredshow.argmax(0)
                # maskpredshow = maskpredshow-maskpredshow.min()
                # if maskpredshow.max()>1:
                    # maskpredshow = maskpredshow/maskpredshow.max()
                # maskpredshow[maskpredshow < 0.5] = 0


                outimg = np.zeros((2 * img[n].shape[1], img[n].shape[2], 3))
                # Left side is input image
                outimg[:img[n].shape[1], :, :] = imgshow
                # Right side is Mask - red is predict, blue is og
                outimg[img[n].shape[1]:, :, 2] = maskshow
                # outimg[img[n].shape[1]:, :, 0] = maskpredshow
                outimg[img[n].shape[1]:, :, 0] = class_pred

                # dice += dice_coeff(maskshow,class_pred)
                # n+=1

                outfn = os.path.join(os.path.dirname(outcsv), os.path.basename(pathModel)[:-4], os.path.basename(imgfn[n]))
                if not os.path.exists(os.path.dirname(outfn)):
                    os.mkdir(os.path.dirname(outfn))
                plt.imsave(outfn, outimg)
        with open(outcsv, 'a') as fd:
            fd.write(pathModel + ',' + str(float(outLoss))+ ',' + str(float(dice/num)) + '\n')


    def inference(tst_fn, pathModel, nn_arch, nn_classes,  trBatchSize, transResize, transCrop,outdir):

        cudnn.benchmark = True
        device = torch.device("cuda:0")

        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nn_arch == 'deeplabv3_resnet101':
            model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=nn_classes, pretrained=False)

        model = torch.nn.DataParallel(model,device_ids=[0,1,2]) #[0,1,2]
        model.to(device)

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        datasetTest = DatasetGenerator(tst_fn, inference=True)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=0, shuffle=False,
                                    pin_memory=True)

        model.eval()
        for i, (img, imgfn) in enumerate(dataLoaderTest):
            seg_probs_cuda = model.forward(img.cuda())
            seg_probs = seg_probs_cuda['out'].cpu()

            for n in range(0,len(imgfn)):
                outimg = np.zeros((2*img[n].shape[1], img[n].shape[2],3))
                base = os.path.join(outdir, os.path.basename(imgfn[n]).split('.')[0])
                outseg = base + '.npy'
                outfig = base + '.png'

                # Original Image
                imgshow = img[n].permute(1, 2, 0).detach().numpy()
                imgshow = imgshow-imgshow.min()
                imgshow = imgshow/imgshow.max()

                # Mask prediction
                maskpredshow = seg_probs[n].detach().numpy()
                # maskpredshow = maskpredshow - maskpredshow.min()
                # maskpredshow = maskpredshow/maskpredshow.max()
                class_pred = maskpredshow.argmax(0)

                # Zero pad from 224->256
                if class_pred.shape==(224,224):
                    m=nn.ZeroPad2d(int((256-224)/2))
                    np.save(outseg,m(torch.from_numpy(class_pred)).numpy())

                # Apply segmentation to original image
                # Transparency in alpha channel (4)
                segd = np.zeros((224, 224, 4))
                segd[:, :, :3] = imgshow
                transparency = maskpredshow[1]/maskpredshow[1].max()
                segd[:, :, 3] = transparency

                # Apply give some weight to non segmented
                # Threshold transparency in alpha channel (4)
                segdt = np.zeros((224, 224, 4))
                segdt[:, :, :3] = imgshow
                transparency = maskpredshow[1]/maskpredshow[1].max()
                transparency[transparency < 0.5] = 0.5
                segdt[:, :, 3] = transparency

                # Instantiate figure
                # fig = plt.figure()
                #
                # # Initialize Plots
                # ax1 = fig.add_subplot(221)
                # ax2 = fig.add_subplot(222)
                # ax3 = fig.add_subplot(211)
                # ax4 = fig.add_subplot(212)
                #
                # # Hide axis
                # ax1.set_axis_off()
                # ax2.set_axis_off()
                # ax3.set_axis_off()
                # ax4.set_axis_off()
                #
                # # Set Title
                # ax1.title.set_text('0_min=' + str(round(maskpredshow[0].min(),2)) + '_max=' + str(round(maskpredshow[0].max(),2)))
                # ax2.title.set_text('1_min=' + str(round(maskpredshow[1].min(),2)) + '_max=' + str(round(maskpredshow[1].max(),2)))
                # ax1.imshow(maskpredshow[0], cmap='gray', vmin=-0, vmax=1)
                # ax2.imshow(maskpredshow[1], cmap='gray', vmin=-0, vmax=1)
                # ax3.imshow(imgshow)
                # ax4.imshow(class_pred, cmap='gray', vmin=-0, vmax=1)
                # fig.savefig(outfig)
                # plt.close()

                fig,axs=plt.subplots(3,2)
                plt.tight_layout()
                axs.flat[0].set_axis_off()
                axs.flat[1].set_axis_off()
                axs.flat[2].set_axis_off()
                axs.flat[3].set_axis_off()
                axs.flat[4].set_axis_off()
                axs.flat[5].set_axis_off()

                axs.flat[0].title.set_text('0_min=' + str(round(maskpredshow[0].min(), 2)) + '_max=' + str(round(maskpredshow[0].max(), 2)))
                axs.flat[0].imshow(maskpredshow[0], cmap='gray', vmin=-0, vmax=1)
                axs.flat[1].title.set_text('1_min=' + str(round(maskpredshow[1].min(), 2)) + '_max=' + str(round(maskpredshow[1].max(), 2)))
                axs.flat[1].imshow(maskpredshow[1], cmap='gray', vmin=-0, vmax=1)
                axs.flat[2].title.set_text('Original Image')
                axs.flat[2].imshow(imgshow)
                axs.flat[3].title.set_text('Segmentation')
                axs.flat[3].imshow(class_pred,cmap='gray',vmin=0,vmax=1)
                axs.flat[4].title.set_text('Segmented Original')
                axs.flat[4].imshow(segd, cmap='gray', vmin=0, vmax=1)
                axs.flat[5].title.set_text('Segmented Transparent')
                axs.flat[5].imshow(segdt, cmap='gray', vmin=0, vmax=1)
                fig.savefig(outfig)
                plt.close()
                #
                # maskpredshow = maskpredshow[1]
                # maskpredshow = maskpredshow-maskpredshow.min()
                # maskpredshow = maskpredshow/maskpredshow.max()
                #
                # outimg[:img[n].shape[1], :, :] = imgshow
                # # maskpredshow[maskpredshow < 0.5] = 0
                # outimg[img[n].shape[1]:, :, 1] = maskpredshow
                # plt.imsave(outfn, outimg)
                #
                # fn_out = os.path.join(outdir,os.path.basename(imgfn[n]).split('.')[0]+'.npy')
                #
                # np.save(fn_out, seg_probs[n].detach().numpy())

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    categorical = np.moveaxis(categorical,-1,1)
    return categorical
