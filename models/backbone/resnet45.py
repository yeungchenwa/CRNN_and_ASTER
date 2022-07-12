import torch
import torch.nn as nn
import torchvision

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def init_parameters(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")         # xavier?
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

class BasicBlock(nn.Module):

  def __init__(self, in_channels, channels, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv1x1(in_channels, channels, stride)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(channels, channels)
    self.bn2 = nn.BatchNorm2d(channels)
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

class ResNet_in_STR(nn.Module):
  """
  The ResNet for Scene Text Recognition
  The whole network means ResNet45, but it is different from standard ResNet45
  """

  def __init__(self):
    super(ResNet_in_STR, self).__init__()

    in_channels = 3
    # The whole network means ResNet45
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    # 3 block1
    self.layer1 = self._make_layer(32,  3, [2, 2])      # [16, 50]
    # 4 block2
    self.layer2 = self._make_layer(64,  4, [2, 2])      # [8, 25]
    # 6 block3
    self.layer3 = self._make_layer(128, 6, [2, 1])      # [4, 25]
    # 6 block4
    self.layer4 = self._make_layer(256, 6, [2, 1])      # [2, 25]
    # 3 block5
    self.layer5 = self._make_layer(512, 3, [2, 1])      # [1, 25]

    for m in self.modules():
        init_parameters(m)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:     # downsample process
      downsample = nn.Sequential(                       # downsample through 1x1 convolution
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(BasicBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4) # [B, C, H, W] [B, 3, 1, 25]

    # Necessary???
    cnn_feat = x5.squeeze(2) # [N, c, w]
    cnn_feat = cnn_feat.transpose(2, 1)
    # print(cnn_feat.size())
    return cnn_feat

"""
x = torch.randn(3, 1, 32, 100)
net = ResNet_in_STR()
feature_map = net(x)
print(feature_map.size())
print(net)
"""