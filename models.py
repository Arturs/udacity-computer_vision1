## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
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
                
# 7 epochs
#         self.conv1_1 = nn.Conv2d(1, 32, 5)
#         self.pool1_1 = nn.MaxPool2d(2, 2)
#         self.fc1_1 = nn.Linear(110*110*32, 136)
        
 
#  35 epochs, didnt improve during the last 2 epochs
#         self.conv2_1 = nn.Conv2d(1, 32, 5)
#         self.pool2_1 = nn.MaxPool2d(2, 2)
#         self.batch_norm2_1 = nn.BatchNorm1d(110*110*32)
#         self.fc2_1 = nn.Linear(110*110*32, 136)
        

#         10 epochs, didnt improve during the last 2 epochs
#         self.conv3_1 = nn.Conv2d(1, 32, 5)
#         self.pool3_1 = nn.MaxPool2d(2, 2)
        
#         self.conv3_2 = nn.Conv2d(32, 64, 5)
#         self.pool3_2 = nn.MaxPool2d(2, 2)
        
#         self.batch_norm3_1 = nn.BatchNorm1d(53*53*64)
#         self.fc3_1 = nn.Linear(53*53*64, 136)
    

#  6 epochs, 
#         self.conv4_1 = nn.Conv2d(1, 32, 5)
#         self.pool4_1 = nn.MaxPool2d(2, 2)       
#         self.conv4_1_drop = nn.Dropout2d()
 
#         self.conv4_2 = nn.Conv2d(32, 64, 5)
#         self.pool4_2 = nn.MaxPool2d(2, 2)
#         self.conv4_2_drop = nn.Dropout2d()
               
#         self.batch_norm4_1 = nn.BatchNorm1d(53*53*64)
#         self.fc4_1 = nn.Linear(53*53*64, 136)
        
                
#  6 epochs      
#         self.conv5_1 = nn.Conv2d(1, 32, 5)
#         self.pool5_1 = nn.MaxPool2d(2, 2)       
#         self.conv5_1_drop = nn.Dropout2d()
 
#         self.conv5_2 = nn.Conv2d(32, 64, 5)
#         self.pool5_2 = nn.MaxPool2d(2, 2)
#         self.conv5_2_drop = nn.Dropout2d()
               
#         self.batch_norm5_1 = nn.BatchNorm1d(53*53*64)
#         self.fc5_1 = nn.Linear(53*53*64, 1024)    
       
#         self.batch_norm5_2 = nn.BatchNorm2d(53*53*64)
#         self.fc5_2 = nn.Linear(1024, 136)   
    
        
        self.conv6_1 = nn.Conv2d(1, 32, 5)
        self.pool6_1 = nn.MaxPool2d(2, 2)       
        self.conv6_1_drop = nn.Dropout2d()
 
        self.conv6_2 = nn.Conv2d(32, 64, 5)
        self.pool6_2 = nn.MaxPool2d(2, 2)
        self.conv6_2_drop = nn.Dropout2d()
     
        self.conv6_3 = nn.Conv2d(64, 128, 5)
        self.pool6_3 = nn.MaxPool2d(2, 2)       
        self.conv6_3_drop = nn.Dropout2d()
     
        self.batch_norm6_1 = nn.BatchNorm1d(24*24*128)
        self.fc6_1 = nn.Linear(24*24*128, 1024)    
   
        self.fc6_2 = nn.Linear(1024, 136)   
        
        
    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        
# model1
#         x = Variable(x)
#         x = self.pool1_1(F.relu(self.conv1_1(x)))
#         x = x.view(-1, 110*110*32)
#         x = self.fc1_1(x)

# model2
#         x = Variable(x)
#         x = self.pool2_1(F.relu(self.conv2_1(x)))
#         x = x.view(-1, 110*110*32)
#         x = self.batch_norm2_1(x)
#         x = self.fc2_1(x)


# model3
#         x = Variable(x)
#         x = self.pool3_1(F.relu(self.conv3_1(x)))
#         x = self.pool3_2(F.relu(self.conv3_2(x)))

#         x = x.view(-1, 53*53*64)
#         x = self.batch_norm3_1(x)
#         x = self.fc3_1(x)


# model4
#         x = Variable(x)
#         x = self.pool4_1(F.relu(self.conv4_1_drop(self.conv4_1(x))))
#         x = self.pool4_2(F.relu(self.conv4_2_drop(self.conv4_2(x))))

#         x = self.pool4_1(F.relu(self.conv4_1(x)))
#         x = self.pool4_2(F.relu(self.conv4_2(x)))
    
#         x = x.view(-1, 53*53*64)
#         x = self.batch_norm4_1(x)
#         x = self.fc4_1(x)

# model5
#         x = Variable(x)
#         x = self.pool5_1(F.relu(self.conv5_1_drop(self.conv5_1(x))))
#         x = self.pool5_2(F.relu(self.conv5_2_drop(self.conv5_2(x))))

#         x = self.pool5_1(F.relu(self.conv5_1(x)))
#         x = self.pool5_2(F.relu(self.conv5_2(x)))

#         x = x.view(-1, 53*53*64)
#         x = self.batch_norm5_1(x)
#         x = self.fc5_1(x)
#         x = self.fc5_2(x)
        
       
#         x = Variable(x)
        x = self.pool6_1(F.relu(self.conv6_1_drop(self.conv6_1(x))))
        x = self.pool6_2(F.relu(self.conv6_2_drop(self.conv6_2(x))))
        x = self.pool6_3(F.relu(self.conv6_3_drop(self.conv6_3(x))))
        
#         x = self.pool6_1(F.relu(self.conv6_1(x)))
#         x = self.pool6_2(F.relu(self.conv6_2(x)))
#         x = self.pool6_3(F.relu(self.conv6_3(x)))
    
        x = x.view(-1, 24*24*128)
        x = self.batch_norm6_1(x)
        x = self.fc6_1(x)
        x = self.fc6_2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
