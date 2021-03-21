#!/usr/bin/env python
# coding: utf-8

# In[21]:


import torch.nn as nn
import torch
from vocab import *
import torchvision.models as models
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

# In[2]:

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

backbone = 'resnet18'
dropout = 0.5
pretrained = True
version = 'v2'

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stacking
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MiCTBlock(nn.Module):
    """
    The MiCTBlock groups all ResNet basic/bottleneck blocks at
    a given feature depth. It performs a parallel 3D convolution
    on the input and then merges the output with the output of
    the first 2D CNN block using point-wise summation to form
    a residual cross-domain connection.
    """
    def __init__(self, block, inplanes, planes, blocks, stride=(1, 1)):
        """
        :param block: the block class, either BasicBlock or Bottleneck.
        :param inplanes: the number of input plances.
        :param planes: the number of output planes.
        :param blocks: the number of blocks.
        :param stride: (temporal, spatial) stride.
        """
        super(MiCTBlock, self).__init__()
        downsample = None
        if stride[1] != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.blocks = blocks
        self.stride = stride
        self.bottlenecks = nn.ModuleList()
        self.bottlenecks.append(block(inplanes, planes, self.stride[1],
                                      downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, self.blocks):
            self.bottlenecks.append(block(self.inplanes, planes))

        self.conv = nn.Conv3d(inplanes, planes, kernel_size=3,
                              stride=(self.stride[0], self.stride[1], self.stride[1]),
                              padding=0, bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = F.pad(x, (1, 1, 1, 1, 0, 2), 'constant', 0)
        out1 = self.conv(out1)
        out1 = self.bn(out1)
        out1 = self.relu(out1)

        x, depth = _to_4d_tensor(x, depth_stride=self.stride[0])
        out2 = self.bottlenecks[0](x)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2

        out, depth = _to_4d_tensor(out)
        for i in range(1, self.blocks):
            out = self.bottlenecks[i](out)
        out = _to_5d_tensor(out, depth)

        return out

class MiCTResNet(nn.Module):
    """
    MiCTResNet is a ResNet backbone augmented with five 3D cross-domain
    residual convolutions.
    The model operates on 5D tensors but since 2D CNNs expect 4D input,
    the data is transformed many times to 4D and then transformed back
    to 5D when necessary. For efficiency only one 2D convolution is
    performed for each kernel by vertically stacking the features maps
    of each video clip contained in the batch.
    This models is inspired from the work by Y. Zhou, X. Sun, Z-J Zha
    and W. Zeng: MiCT: Mixed 3D/2D Convolutional Tube for Human Action
    Recognition.
    """

    def __init__(self, block, layers, dropout, version, embedding_size):
        """
        :param block: the block class, either BasicBlock or Bottleneck.
        :param layers: the number of blocks for each for each of the
            four feature depth.
        :param dropout: dropout rate applied during training.
        :param embedding_size: the embedding size
        """
#         super(MiCTResNet, self).__init() might fix later
        super().__init__()

        self.inplanes = 64
        self.dropout = dropout
        self.version = version
        self.t_strides = {'v1': [1, 1, 2, 2, 2], 'v2': [1, 1, 1, 2, 1]}[self.version]
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)

        self.conv2 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7),
                               stride=(self.t_strides[0], 2, 2),
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = MiCTBlock(block, self.inplanes, 64, layers[0],
                                stride=(self.t_strides[1], 1))
        self.layer2 = MiCTBlock(block, self.layer1.inplanes, 128, layers[1],
                                stride=(self.t_strides[2], 2))
        self.layer3 = MiCTBlock(block, self.layer2.inplanes, 256, layers[2],
                                stride=(self.t_strides[3], 2))
        self.layer4 = MiCTBlock(block, self.layer3.inplanes, 512, layers[3],
                                stride=(self.t_strides[4], 2))

        self.avgpool1 = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout3d(self.dropout)
#         self.fc = nn.Linear(512 * block.expansion, self.n_classes)
        self.linear = nn.Linear(512 * block.expansion, self.embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def transfer_weights(self, state_dict):
        """
        Transfers ResNet weights pre-trained on the ImageNet dataset.
        :param state_dict: the state dictionary of the loaded ResNet model.
        :return: None
        """
        for key in state_dict.keys():
            if key.startswith('conv1') | key.startswith('bn1'):
                eval('self.' + key + '.data.copy_(state_dict[\'' + key + '\'])')
            if key.startswith('layer'):
                var = key.split('.')
                if var[2] == 'downsample':
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].downsample[' + var[3] + '].' +
                         var[4] + '.data.copy_(state_dict[\'' + key + '\'])')
                else:
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].' + var[2] + '.' + var[3] +
                         '.data.copy_(state_dict[\'' + key + '\'])')

    def forward(self, x):
        x = x.view(1,3,20,128,128)
        out1 = F.pad(x, (3, 3, 3, 3, 0, 6), 'constant', 0)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool2(out1)

        x, depth = _to_4d_tensor(x, depth_stride=2)
        out2 = self.conv1(x)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        out2 = self.maxpool1(out2)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2
        # pass through the middle 4 layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.drop(out)
        out = self.avgpool1(out)
        out = out.squeeze(4).squeeze(3)
        
        # input embedding
        out_linear = []
        for i in range(out.size()[-1]):
            out_linear.append(self.linear(out[:, :, i]).unsqueeze(2))
        out_linear = torch.cat(out_linear, 2)
        out = self.avgpool2(out_linear).squeeze(2)        
#         out = self.linear(out)

        return out

class encoder(nn.Module):
    """
        Defines the encoder for the image captioning task
    """
    def __init__(self, embedding_size):
        """
            Initialize the experiment name
        """
        super(encoder, self).__init__()
#         self.experiment_name = experiment_name
        res50_model = models.resnet50(pretrained=True)
        #get all the layers of resnet50
        layers = list(res50_model.children())
        # Removing the last layer 
        layers = layers[:-1]
        self.resnet50_model = nn.Sequential(*layers)
        #replacing the last layer with linear layer
        self.linear = nn.Linear(res50_model.fc.in_features, embedding_size)
        self.batchNorm = nn.BatchNorm1d(embedding_size, momentum=0.01)
    
    def forward(self, x):
        """
           forward pass computation
        """
        #print('shape of input to forward',x.size())
        with torch.no_grad():
            x1 = self.resnet50_model(x)
        #print('shape of output from resnet',x1.size())
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.linear(x1)
        x1 = self.batchNorm(x1)
        return x1
    
# In[3]:


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 100):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(0)
#         print("x.shape: {}".format(x.shape))
#         print("seq_len: {}".format(seq_len))
#         print("self.pe.shape: {}".format(self.pe[:,:seq_len]))
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x


# In[25]:


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


# In[4]:


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

    # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output


# In[5]:


def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
#         print("mask.shape: {}".format(mask.shape))
#         print("scores.shape: {}".format(scores.shape))
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output


# In[6]:


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# In[7]:


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))         / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# In[23]:


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# In[18]:


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = encoder(d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

# Encoder for MiCT-net
class Encoder_MiCT(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        if backbone == 'resnet18':
            self.embed = MiCTResNet(BasicBlock, [2, 2, 2, 2], dropout, version, d_model) # this should be the MiCT-net
            if pretrained:
                self.embed.transfer_weights(model_zoo.load_url(model_urls['resnet18']))
        elif backbone == 'resnet34':
            self.embed = MiCTResNet(BasicBlock, [3, 4, 6, 3], dropout, version, d_model)
            if pretrained:
                self.embed.transfer_weights(model_zoo.load_url(model_urls['resnet34']))     

        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


# In[19]:


class Transformer_mict(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads):
        super().__init__()
#         self.encoder = Encoder(d_model, N, heads)
        self.encoder = Encoder_MiCT(d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
# we don't perform softmax on the output as this will be handled 
# automatically by our loss function


# In[ ]:





# In[ ]:




