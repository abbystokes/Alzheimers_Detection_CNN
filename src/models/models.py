import torch
import torch.nn as nn
import torch.nn.functional as F

class SNeurodCNN(nn.Module):
    def __init__(self):
        super(SNeurodCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((44, 44))
        self.fc1 = nn.Linear(64 * 44 * 44, 500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 3)
        
    def forward(self, x):
        # print(f"{x.size()}")
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 44 * 44) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x