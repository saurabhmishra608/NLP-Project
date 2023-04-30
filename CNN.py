# Importing the required packages
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score
import seaborn as sns
import numpy as np

class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.Conv1 = nn.Sequential(
        nn.Conv2d(1, 32, 5), # 220, 220
        nn.MaxPool2d(2), # 110, 110
        nn.ReLU(),
        nn.BatchNorm2d(32)
        )
        self.Conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 5), # 106, 106
        nn.MaxPool2d(2),  # 53,53
        nn.ReLU(),
        nn.BatchNorm2d(64)
        )
        self.Conv3 = nn.Sequential(
        nn.Conv2d(64, 128, 3), # 51, 51
        nn.MaxPool2d(2), # 25, 25
        nn.ReLU(),
        nn.BatchNorm2d(128)
        )
        self.Conv4 = nn.Sequential(
        nn.Conv2d(128, 256, 3), # 23, 23
        nn.MaxPool2d(2), # 11, 11
        nn.ReLU(),            
        nn.BatchNorm2d(256)
        )
        self.Conv5 = nn.Sequential(
        nn.Conv2d(256, 512, 3), # 9, 9
        nn.MaxPool2d(2), # 4, 4
        nn.ReLU(),
        nn.BatchNorm2d(512)
        )
        
        self.Linear1 = nn.Linear(512 * 4 * 4, 256)
        self.dropout=nn.Dropout(0.1)
        self.Linear3 = nn.Linear(256, 25)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x=self.dropout(x)
        x = self.Conv5(x)
        x = x.view(x.size(0), -1)
        x = self.Linear1(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 75, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(75),
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(75, 50, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 25, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(25),
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(625, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

