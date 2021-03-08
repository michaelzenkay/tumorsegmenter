import os
import numpy as np
import PIL
from PIL import Image
import pandas
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2

#-------------------------------------------------------------------------------- 

class DatasetGenerator(Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathDatasetFile, augment=False, inference=False):

        # Initialize
        self.listImagePaths = []
        self.listSegPaths = []
        self.listImageLabels = []
        self.inference = inference
        self.augment = augment

        #---- Grab image and seg paths

        if isinstance(pathDatasetFile, list):
            frame = []
            for i in range(0, len(pathDatasetFile)):
                df = pandas.read_csv(pathDatasetFile[i])
                frame.append(df)
            csv = pandas.concat(frame, axis=0, ignore_index=True)

        else:
            csv = pandas.read_csv(pathDatasetFile)
            
        for entry in csv.values:
            self.listImagePaths.append(str(entry[0]))

            if self.inference==False:
                segPath=str(entry[1])
                self.listSegPaths.append(segPath)
    
    #-------------------------------------------------------------------------------- 
    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(256, 256),interpolation=2)
        image = resize(image)
        resize = transforms.Resize(size=(256, 256), interpolation=0)
        mask = resize(mask)

        if self.augment==True:
            # Random zoom crop
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(128/2, 128/2))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                resize = transforms.Resize(size=(256, 256), interpolation=2)
                image = resize(image)
                resize = transforms.Resize(size=(256, 256), interpolation=0)
                mask = resize(mask)

            # Random rotate
            if random.random() > 0.5:
                min_angle = -15
                max_angle = 15
                angle = random.randrange(min_angle,max_angle)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask.convert('RGB'), angle,resample=PIL.Image.NEAREST).convert('L')

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(224, 224))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        else:
            image = TF.center_crop(image, 224)
            mask = TF.center_crop(mask, 224)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Normalize
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
        return image, mask
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')

        # Training and Validation
        if self.inference == False:
            segPath = self.listSegPaths[index]
            segData = Image.fromarray(np.load(segPath).astype('uint8'))

            # Transform
            imageData,segData = self.transform(imageData,segData)
            segData[segData > 0] = 1
        
            return imageData, segData, imagePath, segPath

        # Inference
        else:
            imageData, null = self.transform(imageData, imageData.convert('L'))
            return imageData, imagePath
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)