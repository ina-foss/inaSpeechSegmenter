'''
"""
ResNet Implementation in PyTorch
---------------------------------
This file implements variants of the ResNet (Residual Network) architecture using PyTorch,
an open source deep learning framework. ResNet is a widely used neural network architecture
that employs residual connections to allow training of very deep networks by learning residual functions.
The implementation includes both BasicBlock and Bottleneck modules for constructing the network,
and provides a factory function (ResNet101) to create a ResNet101 model for generating embeddings
from input features.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class BasicBlock(nn.Module):
    """
    BasicBlock: A fundamental building block for ResNet architectures.
    It comprises two 3x3 convolutional layers with batch normalization and ReLU activations,
    along with a shortcut connection that adds the input directly to the output to facilitate training.
    
    - Convolutional layers: These layers apply learned filters that slide over the input data to extract local features,
      such as edges or textures, enabling the network to capture spatial hierarchies.
    - ReLU (Rectified Linear Unit): A non-linear activation function that outputs the input directly if it is positive,
      otherwise it outputs zero. This introduces non-linearity into the model while helping to mitigate the vanishing gradient problem.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        Forward Pass:
        Executes the residual learning step for this convolutional block, processing the input by applying a convolution followed 
        by batch normalization and a ReLU activation. 
        Enables the network to learn residual mappings, which improve training stability and performance in deep architectures.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck:
    A more complex ResNet block designed for deeper architectures that reduces computation through a bottleneck structure.
    It uses three convolutional layers: a 1x1 convolution to reduce dimensions, a 3x3 convolution to process features, and a 1x1 convolution to restore the channel size (expansion). 
    A shortcut connection bypasses these layers to enable residual learning, which facilitates the training of very deep networks.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        # self.se = SELayer(planes * 4, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        Forward Pass:
        Defines the forward computation of the Bottleneck block in PyTorch. This method specifies how the input tensor is processed by 
        the sequence of convolutional layers, batch normalization, and ReLU activations, and how the shortcut (residual) connection is applied.
        In PyTorch, the forward method is invoked automatically during model execution to compute the output from the input.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet:
    A customizable residual network built with PyTorch that constructs deep convolutional architectures using either BasicBlock or Bottleneck modules.
    This model processes input features (e.g., spectrograms) to generate fixed-dimensional embeddings by leveraging residual connections for effective deep learning and global pooling to summarize spatial information.
    In the context of audio files, 'spatial information' refers to the patterns and structures present in the time-frequency representation (spectrogram) of the audio signal, capturing how frequency components evolve over time.
    """
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128, squeeze_excitation=False):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.squeeze_excitation = squeeze_excitation
        if block is BasicBlock:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
            current_freq_dim = int((feat_dim - 1) / 2) + 1
            self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.embedding = nn.Linear(m_channels * 8 * 2 * current_freq_dim, embed_dim)
        elif block is Bottleneck:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)
            self.embedding = nn.Linear(int(feat_dim / 8) * m_channels * 16 * block.expansion, embed_dim)
        else:
            raise ValueError(f'Unexpected class {type(block)}.')

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Constructs a sequential layer by stacking multiple residual blocks with specified strides.
        The first block uses the provided stride while subsequent blocks use stride 1, ensuring proper dimensionality,
        and updating the input channel size for subsequent layers based on the block's expansion.
        This function is used to build a stage of the ResNet model, enabling hierarchical feature extraction from input data.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward Pass:
        Processes the input by adding a channel dimension and passing it through the initial convolution and four residual layers.
        It then performs global pooling by computing the mean and standard deviation across the time dimension, concatenates these features,
        and passes them through a linear embedding layer to produce the final fixed-dimensional output embedding.
        This function is used to extract robust embeddings from audio inputs, enabling downstream tasks such as speaker recognition and speech segmentation.
        """
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        pooling_mean = torch.mean(out, dim=-1)
        meansq = torch.mean(out * out, dim=-1)
        pooling_std = torch.sqrt(meansq - pooling_mean ** 2 + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)

        embedding = self.embedding(out)
        return embedding


def ResNet101(feat_dim, embed_dim, squeeze_excitation=False):
    """
    ResNet101:
    A factory function that creates a ResNet101 model using Bottleneck blocks with a configuration of [3, 4, 23, 3] layers.
    It is used to generate deep embeddings from audio inputs, where 'feat_dim' is the feature dimension and 'embed_dim' is the output embedding size.
    These embeddings capture high-level audio features such as spectral shapes, temporal dynamics, speaker timbre, intonation, and phonetic content,
    which are critical for tasks like speaker recognition, speech segmentation, and audio classification.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], feat_dim=feat_dim, embed_dim=embed_dim,
                  squeeze_excitation=squeeze_excitation)
