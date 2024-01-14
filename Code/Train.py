#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision import transforms as tf

from torch.optim import AdamW, SGD
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor, v2
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
# import Misc.ImageUtils as iu
from Network.Network import * 
from Misc.MiscUtils import *
from Misc.DataUtils import *
import ipdb
import wandb


# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="CIFAR10_mymodel",
#     name=f"my_model_s4", 
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.001,
#     "architecture": "MyModel",
#     "dataset": "CIFAR-100",
#     "batch size": 30,
#     "epochs": 50,
#     "loss_function": "CrossEntropyLoss",
#     "optimizer": "SGD",
#     }
# )

device  =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    TBatch = []
    TLabel = []

    VBatch = []
    VLabel = []

    train_percent = 0.8

    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        # for i in range(6):
        #     RandIdx = random.randint(0, len(TrainSet)-1)
            
            
        #     ##########################################################
        #     # Add any standardization or data augmentation here!
        #     ##########################################################
        transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomPerspective(distortion_scale=0.3, p=0.2),
        v2.RandomRotation(degrees=(-30, 30), interpolation=v2.InterpolationMode.BILINEAR, expand=True,fill=255),
        tf.Resize([int(32), int(32)]),
        ])

        RandIdx = random.randint(0, len(TrainSet)-1)

        I1, Label = TrainSet[RandIdx]
        
        I1 = transforms(I1)
        standardized_image = I1.numpy()
        mean = np.mean(standardized_image, axis=(1,2), keepdims=True)
        std = np.std(standardized_image, axis=(1,2), keepdims=True)
        standardized_image = (standardized_image - mean) / (std + 0.0001) 
        standardized_image = torch.from_numpy(standardized_image)
        #standardized_image = I1
        # Label = convertToOneHot(Label, 10)
        # ipdb.set_trace()
        # standardized_image = standardized_image.numpy()
        # plt.subplots(1, 3, figsize=(2,6))

        # for index in range(3):
        #     plt.subplot(1, 3, index+1)
        #     plt.axis('off')
        #     plt.imshow(standardized_image[index])
        # plt.show()



        # Append All Images and Mask
        if ImageNum < MiniBatchSize*train_percent:
            TBatch.append(standardized_image)
            TLabel.append(torch.tensor(Label))

        else :
            VBatch.append(standardized_image)
            VLabel.append(torch.tensor(Label))


        ImageNum += 1

    TBatch, VBatch = torch.stack(TBatch).to(device), torch.stack(VBatch).to(device)
    TLabel, VLabel = torch.stack(TLabel).to(device), torch.stack(VLabel).to(device)

    # ipdb.set_trace()   
    return TBatch, VBatch, TLabel, VLabel



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath):
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    # model = CIFAR10Model(InputSize=3*32*32,OutputSize=10).to(device)
    # model_name = "mymodel_s3"
    # model = ResNet18().to(device)
    # model_name = "ResNet18"
    # model = ResNeXt29_32x4d().to(device)
    # model_name = "ResNeXt"
    model = DenseNet121().to(device)
    model_name = "DenseNet121"
    LogsPath = LogsPath + "/" + model_name


    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = SGD(model.parameters(),lr=0.001,  momentum=0.9, weight_decay=5e-4)


    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)



    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + model_name + "/" + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
        
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        loss_iter = []
        accuracy_iter = []
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            TBatch, VBatch, TLabel, VLabel  = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            
            # Predict output with forward pass
            LossThisBatch = model.training_step(TBatch , TLabel)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            result, output, labels = model.validation_step(VBatch , VLabel)
            
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath  + "/" + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                # ipdb.set_trace()
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                print('\n' + SaveName + ' Model Saved...')
                model.epoch_end(Epochs*NumIterationsPerEpoch + PerEpochCounter, result)


            # Log
            loss_iter.append(result["loss"].cpu())
            accuracy_iter.append(result["acc"].cpu())
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # wandb.log({"Accuracy_per_batch": result["acc"], "LossEveryIter_batch": result["loss"]})

            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()


        loss_iter = np.array(loss_iter)
        accuracy_iter = np.array(accuracy_iter)
        Writer.add_scalar('LossEveryEpoch', np.mean(loss_iter), PerEpochCounter)
        Writer.add_scalar('Accuracy_per_epoch', np.mean(accuracy_iter), PerEpochCounter)
        # wandb.log({"Accuracy_per_epoch": np.mean(accuracy_iter), "LossEveryEpoch": np.mean(loss_iter)})

            

        # Save model every epoch
        SaveName = CheckPointPath + model_name + "/" + str(Epochs) + model_name +'.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')
        # scheduler.step()
        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=100, help='Number of Epochs to Train for, Default:20')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=30, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=ToTensor())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = "./content/Checkpoints/"
    LogsPath = "./content/Logs"
    BasePath = ".\CIFAR10\Train"

    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath,CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = "19mymodel2"
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)
    # ipdb.set_trace()
    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath)

    
if __name__ == '__main__':
    main()
 
