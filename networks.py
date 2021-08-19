"""
@Author: Andrés Alejandro Córdova Galleguillos
"""

# Import pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use CUDA if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    """
    Encoder. This has the option of working with GRU or LSTM.
    ! Be consistent! If the encoder was initialize with GRU,
    ! then then the decoder should also be initialized with GRU
    """
    def __init__(self, input_size, hidden_size, recurrent_type="GRU"):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        # It could initialize LSTM or GRU as a recurrent network.
        if recurrent_type == "LSTM":
            self.recurrent = nn.LSTM(hidden_size, hidden_size)
        else:
            self.recurrent = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.recurrent(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initLSTMHidden(self):
        # When working with LSTM, there is the double of hidden states
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))

class DecoderRNN(nn.Module):
    """
    Decoder. It should receive the same hidden states and
    use the same type of recurrent network as the Encoder.
    """
    def __init__(self, hidden_size, output_size, recurrent_type='GRU'):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)

        # It could initialize LSTM or GRU as a recurrent network.
        if recurrent_type == 'LSTM':
            self.recurrent = nn.LSTM(hidden_size, hidden_size)
        else:
            self.recurrent = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    """
    Attention Decoder. It should receive the same hidden states and
    use the same type of recurrent network as the Encoder.
    """
    def __init__(self, hidden_size, output_size, recurrent_type='GRU', dropout_p=0.1, max_length=200):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # Two linear layers for Attention are created.
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        # It could initialize LSTM or GRU as a recurrent network.
        if recurrent_type == 'LSTM':
            self.recurrent = nn.LSTM(self.hidden_size, self.hidden_size)
        else:
            self.recurrent = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # It takes the embedded and the hidden to calculate the attention layers
        # The size of the hidden states differs from GRU to LSTM
        if isinstance(self.recurrent,nn.modules.rnn.LSTM):
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        else:
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # Put the embedding and the attention layers together
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)