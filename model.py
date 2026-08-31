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
        # Apply convolution -> activation function -> pooling
        #x = self.pool(F.relu(self.conv1(x)))
        #print('Input Shape: ', x.shape)
        x = self.batch_norm(x)
        
        x = self.conv1(x)
        #print('Shape after 1st Convolution: ', x.shape)

        x = F.leaky_relu(x)
        x = self.conv2(x)
        #print('Shape after 2nd Convolution: ', x.shape)
        
        x = F.leaky_relu(x)
        x = self.conv3(x)
        #print('Shape after 3rd Convolution: ', x.shape)
        
        x = F.leaky_relu(x)
        x = self.pool(x)
        #print('Shape after Pooling: ', x.shape)

        x = torch.flatten(x, 1) 
        #print('Shape after Flattening: ', x.shape)
        x = self.fc(x)
        #print('Shape after projection layer: ', x.shape)
        x = F.leaky_relu(x)

        if self.num_fc > 1:
            for layer in self.linears:
                x = layer(x)
                x = F.leaky_relu(x)
        
        return x

    def forward(self, x):

        x = self.out(x)
        
        return x

class Readout(nn.Module):
    def __init__(self, input_dim=224, hidden_dim=32, num_classes=4, dropout=0.0, num_heads=1, num_preR=3, preR_dim=32):
        super(Readout, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_preR = num_preR
        self.preR_dim = preR_dim
        self.dropout = 0

        self.num_heads = num_heads
        if self.num_heads > 1:
            self.multi_head = True
        else:
            self.multi_head = False
        
        self.batch_norm = nn.BatchNorm1d(self.preR_dim*self.num_preR, affine=False)  
        self.fc0 = nn.Linear(self.preR_dim*self.num_preR, self.hidden_dim)
        nn.init.kaiming_normal_(self.fc0.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc0.bias)
        
        self.fc1 = nn.Linear(self.num_classes*self.num_preR + self.hidden_dim, self.hidden_dim*2)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(self.hidden_dim*2, self.num_classes)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        if self.multi_head:
            self.attn_embed = nn.Linear(self.num_classes*self.num_preR, self.hidden_dim*self.num_heads)
            self.multihead_attn = nn.MultiheadAttention(self.num_heads*self.hidden_dim, self.num_heads, dropout=self.dropout, batch_first=True)
            self.attn_out = nn.Linear(self.num_heads*self.hidden_dim, self.num_classes*self.num_preR)
        else:
            self.attn_embed = nn.Linear(self.num_classes*self.num_preR, self.hidden_dim)
            self.attn_out = nn.Linear(self.hidden_dim, self.num_classes*self.num_preR)
            
    def input(self, ind_embeds):
        embed = self.batch_norm(ind_embeds)
        embed = self.fc0(embed)
        return embed
        
    def forward(self, logits, ind_embeds):
        x = self.attn_embed(logits)
        q = x
        k = x
        v = x

        if not self.multi_head:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
            x = F.tanh(self.attn_out(x))
            embed = F.tanh(self.input(ind_embeds))

            x = self.fc1(torch.cat([x,embed],dim=1))
            x = F.leaky_relu(x)
            
            return self.fc2(x)
            
        else:
            x, _ = self.multihead_attn(q, k, v, need_weights=False)
            x = F.tanh(self.attn_out(x))
            embed = F.tanh(self.input(ind_embeds))

            x = self.fc1(torch.cat([x,embed],dim=1))
            x = F.leaky_relu(x)
            
            return self.fc2(x)

import math

def conv2d_output_size(input_size, kernel_size, padding=0, stride=1, dilation=1):
    """
    Calculate the output size of a 2D convolution.

    Parameters
    ----------
    input_size : tuple (H, W)
    kernel_size : int or tuple (kH, kW)
    padding : int or tuple (pH, pW)
    stride : int or tuple (sH, sW)
    dilation : int or tuple (dH, dW)

    Returns
    -------
    tuple
        (output_height, output_width)
    """

    H_in, W_in = input_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    kH, kW = kernel_size
    pH, pW = padding
    sH, sW = stride
    dH, dW = dilation

    H_out = math.floor((H_in + 2*pH - dH*(kH - 1) - 1) / sH + 1)
    W_out = math.floor((W_in + 2*pW - dW*(kW - 1) - 1) / sW + 1)

    return H_out, W_out