from unicodedata import bidirectional
import torch
import torch.nn as nn

class DeepbiLSTM(nn.Module):
    """
    Two biLSTM layer to the sequence modeling
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
    """
    def __init__(self, input_size, hidden_size):
        super(DeepbiLSTM, self).__init__()
        self.biLSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        # self.linear = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.biLSTM.flatten_parameters()
        output, _ = self.biLSTM(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # output = self.linear(recurrent)  # batch_size x T x output_size
        return output