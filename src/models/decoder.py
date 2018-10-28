import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import DotProductAttention
from utils.utils import IGNORE_ID, pad_list


class Decoder(nn.Module):
    """
    """

    def __init__(self, vocab_size, embedding_dim, sos_id, eos_id, hidden_size,
                 num_layers, bidirectional_encoder=True, rnn_type='lstm'):
        super(Decoder, self).__init__()
        # Hyper parameters
        # embedding + output
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        # rnn
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder  # useless now
        self.rnn_type = rnn_type
        # Components
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers,
                               batch_first=True,
                               bidirectional=False)
        self.attention = DotProductAttention(hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, padded_input, encoder_padded_outputs):
        """
        Args:
            padded_input: N x To
            # encoder_hidden: (num_layers * num_directions) x N x H
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        # *********Get Input and Output
        # from espnet/Decoder.forward()
        # TODO: need to make more smart way
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        # print("ys_in_pad", ys_in_pad.size())
        assert ys_in_pad.size() == ys_out_pad.size()
        batch_size = ys_in_pad.size(0)
        output_length = ys_in_pad.size(1)
        max_length = ys_in_pad.size(1) - 1  # TODO: should minus 1(sos)?

        # *********Init decoder rnn
        # c_list = [self.zero_state(encoder_padded_outputs)]
        # h_list = [self.zero_state(encoder_padded_outputs)]
        # for l in range(1, self.num_layers):
        #     c_list.append(self.zero_state(encoder_padded_outputs))
        #     h_list.append(self.zero_state(encoder_padded_outputs))

        # *********step decode
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            # step_output is log_softmax()
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)
            #
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((step < lengths) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # *********Run each component
        decoder_input = ys_in_pad
        embedded = self.embedding(decoder_input)
        rnn_output, decoder_hidden = self.rnn(embedded)  # use zero state
        output, attn = self.attention(rnn_output, encoder_padded_outputs)
        output = output.contiguous().view(-1, self.hidden_size)
        predicted_softmax = F.log_softmax(self.out(output), dim=1).view(
            batch_size, output_length, -1)
        for t in range(predicted_softmax.size(1)):
            step_output = predicted_softmax[:, t, :]
            step_attn = attn[:, t, :]
            decode(t, step_output, step_attn)

        return decoder_outputs, decoder_hidden

    # def zero_state(self, encoder_padded_outputs):
    #     N = encoder_padded_outputs.size(0)
    #     return encoder_padded_outputs.new_zeros(N, self.hidden_size)
