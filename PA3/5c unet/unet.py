import torch.nn as nn
import torch

class unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512 + 256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64 + 32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 27, kernel_size=1)

    def forward(self, x):
        # TODO: Modified both values (coefficient of ReLU and the coefficient of x)
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        out_encoder = self.bnd5(self.relu(self.conv5(x4)))
        
        # Call ReLU here
        score = self.relu(out_encoder)
        
        # Implementing transposed convolution layers
        tr1 = self.bn1(self.relu(self.deconv1(score)))
        x4tr1 = torch.cat((x4, tr1), 1)
        tr2 = self.bn2(self.relu(self.deconv2(x4tr1)))
        x3tr2 = torch.cat((x3, tr2), 1)
        tr3 = self.bn3(self.relu(self.deconv3(x3tr2)))
        x2tr3 = torch.cat((x2, tr3), 1)
        tr4 = self.bn4(self.relu(self.deconv4(x2tr3)))
        x1tr4 = torch.cat((x1, tr4), 1)
        out_decoder = self.bn5(self.relu(self.deconv5(x1tr4)))
        
        # Complete the forward function for the rest of the decoder 
        score = self.classifier(out_decoder)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)