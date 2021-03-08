from glob import glob
import os
from os.path import join, basename, dirname
import numpy as np
import time
import sys
from SegNetTrainer import SegNetTrainer

#-------------------------------------------------------------------------------- 

def main():

    arch = 'deeplabv3_resnet101'
    modelfn = '/data/mike/breast/sets/models/deeplab20201031.pth'
    nameCkpt = 'checkpointname'

    ## Initialization from a model
    # init = [None, 'checkpoint','transfer','warmstart']
    init = 'warmstart'

    # Dataset
    pathDataset = '/path/to/dataset'

    # Save
    pathCkpt = pathDataset
    if not os.path.exists(pathCkpt):
        os.mkdir(pathCkpt)
    classes = 2
    batch = 20
    resize = 256
    crop = 224
    folds = 5

    runTrain(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt, modelfn=modelfn, init=init)
    runTest(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt)
    runInference(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt)

#--------------------------------------------------------------------------------   

def runTrain(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt, modelfn=None,init=None,lr = 0.0001, epochs = 100, pretrain = False):
    for f in range(1, folds+1):
        # Dataset
        #---- Paths to the files with training, validation and testing sets.
        #---- Each csv should have rows with [imgfn,segfn,lbl]
        trainfn = join(pathDataset, 'train_' + str(f) +'.csv')
        valfn = join(pathDataset, 'val_' + str(f) + '.csv')    

        # Save
        NameCkpt = nameCkpt + str(f) + '_'

        SegNetTrainer.train(trainfn, valfn, arch, pretrain, classes, batch, epochs, resize, crop, modelfn, init, pathCkpt, NameCkpt, lr)

def runTest(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt):
    testfn = join(pathDataset, 'test.csv')
    for fold in range(1, folds+1):
        modelfns = sorted(glob(join(pathCkpt, nameCkpt + str(fold) + '*.pth')))
        for modelfn in modelfns:
            csv_out = join(dirname(modelfn), 'results_' + str(fold) + '_' + nameCkpt + '.csv')
            SegNetTrainer.test(testfn, modelfn, arch, classes, batch, resize, crop, csv_out)

def runInference(arch, classes, batch, resize, crop, folds, pathDataset, pathCkpt, nameCkpt):
    csv_fn = join(pathDataset, 'test.csv')
    for fold in range(1, folds+1):
        modelfn = sorted(glob(join(pathCkpt, nameCkpt + str(fold) + '*.pth')))[-1]
        csv_out = join(dirname(modelfn), 'results_' + str(fold) + '_' + nameCkpt + '.csv')
        SegNetTrainer.inference(csv_fn, modelfn, arch, classes, batch,
                                resize, crop, dirname(csv_out))
#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()

