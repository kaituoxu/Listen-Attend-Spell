import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    r"""Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               bidirectional=bidirectional)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H 
        """
        # Add total_length for supportting nn.DataParallel() later
        # see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        return output, hidden

    def flatten_parameters(self):
        self.rnn.flatten_parameters()
