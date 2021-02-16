import torch.nn as nn
import torch

class resNetBlock(nn.Module):
    """
        Defines a basic ResNet block that we are going to use for semantic segmentation
    """
    def __init__(self, n):
        """
            Args: n - Number of feature maps in each block
        """
        super(resNetBlock, self).__init__()
        self.nf = n
        self.model = self.build_block(n)
        
    def build_block(self, n):
        """
            Args: n - Number of feature maps in each block
        """
        model = []
        model += 2 * [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n, n, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n),
            nn.LeakyReLU(0.01, inplace=True)
            
        ]
        return nn.Sequential(*model)
    
    def forward(self, x):
        """
            Args : x - Input that needs to be forward passed
        """
        return x + self.model(x)

# Residual network based model for Semantic segmentation
class resUNet(nn.Module):
    """
        Class that denotes a Residual network based semantic segmentation model
    """
    def __init__(self, n_class, n=128, debug_mode=False):
        super().__init__()
        self.n_class = n_class
        self.n = n
        self.debug_mode = debug_mode
        self.block = self.model(n)
        
    def model(self, n):
        """
            Callable to execute the model
        """
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.leakyRelu = nn.LeakyReLU(0.01, inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(64 + 128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(32 + 64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bnd3 = nn.BatchNorm2d(32)

        self.classifier = nn.Conv2d(32, 27, kernel_size=1)

    def forward(self, x):
        """
            Basic model of 
        """
        x1 = self.bn1(self.leakyRelu(self.conv1(x)))
        if(self.debug_mode):
            print("Dimensions after first pass of pad, conv, BN, Relu = {}".format(x1.shape))
        
        x2 = self.bn2(self.leakyRelu(self.conv2(x1)))
        if(self.debug_mode):
            print("Dimensions after second pass of pad, conv, BN, Relu = {}".format(x2.shape))
        
        x3 = self.bn3(self.leakyRelu(self.conv3(x2)))
        if(self.debug_mode):
            print("Dimensions after third pass of pad, conv, BN, Relu = {}".format(x3.shape))
        
        # Short skip connections
        x4 = self.bn4(self.leakyRelu(self.conv4(x3)))
        if(self.debug_mode):
            print("Dimensions after fourth pass of pad, conv, BN, Relu = {}".format(x4.shape))
        x5 = x4 + x3
        
        if(self.debug_mode):
            print("Dimensions after short skip connection 1 = {}".format(x5.shape))
        
        # Short skip connections
        x6 = self.bn4(self.leakyRelu(self.conv4(x5)))
        if(self.debug_mode):
            print("Dimensions after sixth pad, conv, BN, Relu = {}".format(x6.shape))
        x7 = x6 + x5
        if(self.debug_mode):
            print("Dimensions after short skip connection 2 = {}".format(x7.shape))
        
        # Short skip connections
        x8 = self.bn4(self.leakyRelu(self.conv4(x7)))
        if(self.debug_mode):
            print("Dimensions after eighth pass of pad, conv, BN, Relu = {}".format(x8.shape))
        x9 = x8 + x7
        
        if(self.debug_mode):
            print("Dimensions after short skip connection 3 = {}".format(x9.shape))
        
        tr1 = self.bnd1(self.leakyRelu(self.deconv1(x9)))
        if(self.debug_mode):
            print("Dimensions of x3 = {}".format(x3.shape))
            print("Dimensions of tr1 = {}".format(tr1.shape))
            
        # Adding long skip connections
        x2tr1 = torch.cat((x2, tr1), dim=1)
        
        if(self.debug_mode):
            print("Dimensions of tr1 is {}".format(tr1.shape))
            print("Dimensions of x2tr1 is {}".format(x2tr1.shape))
            
        tr2 = self.bnd2(self.leakyRelu(self.deconv2(x2tr1)))
        
        # Adding long skip connections
        x1tr2 = torch.cat((x1, tr2), dim=1)
        
        if(self.debug_mode):
            print("Dimensions of tr2 is {}".format(tr2.shape))
            print("Dimensions of x1tr2 is {}".format(x1tr2.shape))
        
        tr3 = self.bnd3(self.leakyRelu(self.deconv3(x1tr2)))
        
        # Adding long skip connections
        # x2tr3 = torch.cat((x1, tr3), dim=1)
        
        if(self.debug_mode):
            print("Dimensions of tr3 is {}".format(tr3.shape))
        
        output_decoder = self.classifier(tr3)
        
        if(self.debug_mode):
            print("Dimensions of the output_decoder is {}".format(output_decoder.shape))
        
        return output_decoder
