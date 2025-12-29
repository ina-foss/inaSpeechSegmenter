"""
PyTorch implementations of the inaSpeechSegmenter models.

This module contains PyTorch equivalents of the original Keras models used for:
- Speech/Music discrimination (SpeechMusicCNN)
- Speech/Music/Noise discrimination (SpeechMusicNoiseCNN)
- Speaker Gender classification (MaleFemaleCNN)
- Voice Femininity Scoring (Interspeech2023All, Interspeech2023Cvfr)

These architectures are designed to be numerically equivalent to the original 
Keras .hdf5 models when weights are converted using the accompanying conversion scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBase(nn.Module):
    def __init__(self, input_channels=1):
        super(CNNBase, self).__init__()

class SpeechMusicCNN(nn.Module):
    """
    CNN for Speech vs Music classification.
    
    Architecture:
    - 4 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    - Flatten
    - 4 Dense blocks (Linear -> BatchNorm -> ReLU -> Dropout)
    - Output Dense layer (Softmax)
    
    Input: Spectrogram patches (Batch, 1, 68, 21)
    Output: 2 classes (Speech, Music)
    """
    def __init__(self):
        super(SpeechMusicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 4))
        self.bn1 = nn.BatchNorm2d(64, eps=1e-3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(128, eps=1e-3)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128, eps=1e-3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(256, eps=1e-3)
        self.pool3 = nn.MaxPool2d(kernel_size=(12, 1), ceil_mode=True)
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(256, 512)
        self.bn5 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(512, 512)
        self.bn7 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.dense4 = nn.Linear(512, 512)
        self.bn8 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout4 = nn.Dropout(0.5)
        
        self.dense5 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.bn5(self.dense1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn6(self.dense2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn7(self.dense3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn8(self.dense4(x)))
        x = self.dropout4(x)
        x = self.softmax(self.dense5(x))
        return x

class SpeechMusicNoiseCNN(nn.Module):
    """
    CNN for Speech vs Music vs Noise classification.
    
    Architecture:
    - 4 Convolutional blocks
    - Flatten
    - 4 Dense blocks
    - Output Dense layer (Softmax)
    
    Input: Spectrogram patches (Batch, 1, 68, 21)
    Output: 3 classes (Speech, Music, Noise)
    """
    def __init__(self):
        super(SpeechMusicNoiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 4))
        self.bn1 = nn.BatchNorm2d(64, eps=1e-3)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 2), ceil_mode=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(128, eps=1e-3)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128, eps=1e-3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(256, eps=1e-3)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), ceil_mode=True)
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256, eps=1e-3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256, eps=1e-3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.dense4 = nn.Linear(256, 256)
        self.bn8 = nn.BatchNorm1d(256, eps=1e-3)
        self.dropout4 = nn.Dropout(0.5)
        
        self.dense5 = nn.Linear(256, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.bn5(self.dense1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn6(self.dense2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn7(self.dense3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn8(self.dense4(x)))
        x = self.dropout4(x)
        x = self.softmax(self.dense5(x))
        return x

class MaleFemaleCNN(nn.Module):
    """
    CNN for Speaker Gender classification (Male vs Female).
    
    Architecture:
    - 5 Convolutional blocks with varying kernel sizes and pooling
    - Flatten
    - 4 Dense blocks
    - Output Dense layer (Softmax)
    
    Input: Spectrogram patches (Batch, 1, 68, 24)
    Output: 2 classes (Female, Male) - order depends on training labels
    """
    def __init__(self):
        super(MaleFemaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(64, eps=1e-3)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-3)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), ceil_mode=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128, eps=1e-3)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(128, eps=1e-3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(256, eps=1e-3)
        self.pool3 = nn.MaxPool2d(kernel_size=(28, 1), ceil_mode=True)
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(256, 512)
        self.bn6 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(512, 512)
        self.bn7 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(512, 512)
        self.bn8 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.dense4 = nn.Linear(512, 512)
        self.bn9 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout4 = nn.Dropout(0.5)
        
        self.dense5 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.bn6(self.dense1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn7(self.dense2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn8(self.dense3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn9(self.dense4(x)))
        x = self.dropout4(x)
        x = self.softmax(self.dense5(x))
        return x

class Interspeech2023All(nn.Module):
    """
    MLP for Voice Femininity Scoring (General model).
    
    Architecture:
    - Input Dense layer (512 units)
    - Output Dense layer (Sigmoid)
    
    Input: x-vectors (Batch, 256)
    Output: Femininity score (0-1)
    """
    def __init__(self):
        super(Interspeech2023All, self).__init__()
        self.dense1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = self.dropout1(x)
        x = self.sigmoid(self.dense2(x))
        return x

class Interspeech2023Cvfr(nn.Module):
    """
    MLP for Voice Femininity Scoring (CVFR specialized model).
    
    Architecture:
    - 4 Dense blocks (Linear -> BatchNorm -> ReLU -> Dropout)
    - Output Dense layer (Sigmoid)
    
    Input: x-vectors (Batch, 256)
    Output: Femininity score (0-1)
    """
    def __init__(self):
        super(Interspeech2023Cvfr, self).__init__()
        self.dense1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.dense3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.dense4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-3)
        self.dropout4 = nn.Dropout(0.5)
        
        self.dense5 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.dense1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.dense2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.dense3(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.dense4(x)))
        x = self.dropout4(x)
        x = self.sigmoid(self.dense5(x))
        return x
