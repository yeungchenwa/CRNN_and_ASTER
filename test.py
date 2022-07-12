import torch
import yaml
from models.backbone.resnet45 import ResNet_in_STR
from models.sequence.biLSTM import DeepbiLSTM
from models.prediction.CTC import CTC

backbone = ResNet_in_STR()  # [B, 25, 512]
sequence_modeling = DeepbiLSTM(512, 256)
ctc = CTC(256, 30)

x = torch.randn(3, 1, 32, 100)

out = backbone(x)
out = sequence_modeling(out)
out = ctc(out)
print(out.size())
