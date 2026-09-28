import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=224, num_classes=4, kernel_size=4, kernel_stride=4, in_channels=1, conv_channels=8, out_channels=128, padding=0, hidden_dim=128, num_fc=1, num_preR=3):
        super(SimpleCNN, self).__init__()

        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.out_channels = out_channels
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.kernel_stride = kernel_stride
        self.num_preR = num_preR
        self.num_fc = num_fc
        
        self.batch_norm = nn.BatchNorm2d(self.in_channels, affine=False)
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.conv_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.kernel_stride)  
        self.conv2 = nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels//2, kernel_size=self.kernel_size, padding=self.padding, stride=self.kernel_stride)
        self.conv3 = nn.Conv2d(in_channels=self.conv_channels//2, out_channels=self.conv_channels, kernel_size=6, padding=self.padding, stride=self.kernel_stride)
        
        self.pool = lambda x: torch.mean(x, dim=1)
        
        self.fc = nn.Linear(484,self.out_channels)

        if self.num_fc > 1:
            self.linears = nn.ModuleList([nn.Linear(self.out_channels, self.out_channels) for i in range(self.num_fc-1)])
        
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc.bias)
        
        self.out = nn.Linear(self.out_channels*self.num_preR, num_classes)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        
    def embed(self, x):
        x = self.batch_norm(x)
        
        x = self.conv1(x)

        x = F.leaky_relu(x)
        x = self.conv2(x)
        
        x = F.leaky_relu(x)
        x = self.conv3(x)
        
        x = F.leaky_relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1) 

        x = self.fc(x)
        x = F.leaky_relu(x)

        if self.num_fc > 1:
            for layer in self.linears:
                x = layer(x)
                x = F.leaky_relu(x)
        
        return x

    def forward(self, x):

        x = self.out(x)
        
        return x