## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        ##### 224 x 224 input images
        # output size = (W-F)/S +1 = (224 - 5)/1 + 1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        
        # output size = (W-F)/S +1 = (110-4)/1 +1 = 107
        # the output Tensor for one image, will have the dimensions: (64, 107, 107)
        # after one pool layer, this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 4)
        
        # output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output Tensor for one image, will have the dimensions: (128, 51, 51)
        # after one pool layer, this becomes (128, 25, 25) - rounds down
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # output size = (W-F)/S +1 = (25-2)/1 +1 = 24
        # the output Tensor for one image, will have the dimensions: (256, 24, 24)
        # after one pool layer, this becomes (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers 
        # (such as dropout or batch normalization) to avoid overfitting
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        # 256 outputs * the 12*12 filtered/pooled map size = 36864
        self.fc1 = nn.Linear(256*12*12, 1000)
        
        self.fc2 = nn.Linear(1000, 1000)
        
        self.fc3 = nn.Linear(1000, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        
        # Flatten feature maps into a vector
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
