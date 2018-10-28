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
                 num_layers, bidirectional_encoder=True):
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
        self.encoder_hidden_size = hidden_size  # must be equal now
        # Components
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim +
                                 self.encoder_hidden_size, self.hidden_size)]
        for l in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.hidden_size, self.hidden_size)]
        self.attention = DotProductAttention()
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + self.hidden_size,
                      self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.vocab_size))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.hidden_size if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

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
        # max_length = ys_in_pad.size(1) - 1  # TODO: should minus 1(sos)?

        # *********Init decoder rnn
        h_list = [self.zero_state(encoder_padded_outputs)]
        c_list = [self.zero_state(encoder_padded_outputs)]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_padded_outputs))
            c_list.append(self.zero_state(encoder_padded_outputs))
        att_c = self.zero_state(encoder_padded_outputs,
                                H=encoder_padded_outputs.size(2))
        y_all = []

        # **********LAS: 1. decoder rnn 2. attention 3. concate and MLP
        embedded = self.embedding(ys_in_pad)
        for t in range(output_length):
            # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
            rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l-1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]  # below unsqueeze: (N x H) -> (N x 1 x H)
            # step 2. attention: c_i = AttentionContext(s_i,h)
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                          encoder_padded_outputs)
            att_c = att_c.squeeze(dim=1)
            # step 3. concate s_i and c_i, and input to MLP
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)
            y_all.append(predicted_y_t)

        y_all = torch.stack(y_all, dim=1)  # N x To x C
        # **********Cross Entropy Loss
        # F.cross_entropy = NLL(log_softmax(input), target))
        y_all = y_all.view(batch_size * output_length, self.vocab_size)
        ce_loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                  ignore_index=IGNORE_ID,
                                  reduction='elementwise_mean')
        # TODO: should minus 1 here ?
        # ce_loss *= (np.mean([len(y) for y in ys_in]) - 1)
        # print("ys_in\n", ys_in)
        # temp = [len(x) for x in ys_in]
        # print(temp)
        # print(np.mean(temp) - 1)
        return ce_loss

        # *********step decode
        # decoder_outputs = []
        # sequence_symbols = []
        # lengths = np.array([max_length] * batch_size)

        # def decode(step, step_output, step_attn):
        #     # step_output is log_softmax()
        #     decoder_outputs.append(step_output)
        #     symbols = decoder_outputs[-1].topk(1)[1]
        #     sequence_symbols.append(symbols)
        #     #
        #     eos_batches = symbols.data.eq(self.eos_id)
        #     if eos_batches.dim() > 0:
        #         eos_batches = eos_batches.cpu().view(-1).numpy()
        #         update_idx = ((step < lengths) & eos_batches) != 0
        #         lengths[update_idx] = len(sequence_symbols)
        #     return symbols

        # # *********Run each component
        # decoder_input = ys_in_pad
        # embedded = self.embedding(decoder_input)
        # rnn_output, decoder_hidden = self.rnn(embedded)  # use zero state
        # output, attn = self.attention(rnn_output, encoder_padded_outputs)
        # output = output.contiguous().view(-1, self.hidden_size)
        # predicted_softmax = F.log_softmax(self.out(output), dim=1).view(
        #     batch_size, output_length, -1)
        # for t in range(predicted_softmax.size(1)):
        #     step_output = predicted_softmax[:, t, :]
        #     step_attn = attn[:, t, :]
        #     decode(t, step_output, step_attn)
