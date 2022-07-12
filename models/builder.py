import torch
import torch.nn as nn

from models.backbone.resnet45 import ResNet_in_STR
from models.sequence.biLSTM import DeepbiLSTM
from models.prediction.CTC import CTC

class CRNN_builder(nn.Module):
    """
    build the CRNN model
    """
    def __init__(self, feat_channels, hidden_size, num_class):
        super(CRNN_builder, self).__init__()
        self.backbone = ResNet_in_STR()
        self.sequence_modeling = DeepbiLSTM(feat_channels, hidden_size)
        self.ctc = CTC(hidden_size*2, num_class)

    def forward(self, x):
        out = self.backbone(x)
        out = self.sequence_modeling(out)
        out = self.ctc(out)
        return out
