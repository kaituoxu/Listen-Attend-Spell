# Just for learning unittest
# 1. run `. ./path.sh` first
# 2. run `python -m unittest test_encoder.py`
#    or  `python test_encoder.py`
import unittest

import torch

from models.encoder import Encoder


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.input_size = 8
        self.hidden_size = 32
        self.num_layers = 2
        self.bidirectional = True
        self.rnn_type = 'lstm'
        self.N = 4
        self.T = 10
        self.padded_input = torch.randn(self.N, self.T, self.input_size)
        # NOTE: must specify dtype=torch.int
        self.input_lengths = torch.tensor([self.T]*self.N, dtype=torch.int)
        self.padded_input[-2, -2:, ] = 0
        self.input_lengths[-2] = self.T - 2
        self.padded_input[-1, -3:, ] = 0
        self.input_lengths[-1] = self.T - 3

    def test_forward(self):
        encoder = Encoder(self.input_size, self.hidden_size, self.num_layers,
                          bidirectional=self.bidirectional,
                          rnn_type=self.rnn_type)
        output, hidden = encoder(self.padded_input, self.input_lengths)
        self.assertTrue(output.size(), torch.Size(
            [self.N, self.T, self.hidden_size]))


if __name__ == "__main__":
    # uncomment below for unittest
    # unittest.main()

    # Non-unittest part
    input_size = 8
    hidden_size = 5
    num_layers = 2
    bidirectional = True
    rnn_type = 'lstm'
    N = 4
    T = 10
    padded_input = torch.randn(N, T, input_size)
    input_lengths = torch.tensor([T]*N, dtype=torch.int)
    padded_input[-2, -2:, ] = 0
    input_lengths[-2] = T - 2
    padded_input[-1, -3:, ] = 0
    input_lengths[-1] = T - 3

    print(padded_input)
    print(padded_input.size())
    print(input_lengths)
    print(input_lengths.size())
    encoder = Encoder(input_size, hidden_size, num_layers,
                      bidirectional=bidirectional,
                      rnn_type=rnn_type)
    output, hidden = encoder(padded_input, input_lengths)
    print(output.size())
    print(output)
    print(hidden[0].size())
    print(hidden[0])

    import sys
    sys.exit(0)
    # test with data.py
    import json
    from data.data import AudioDataset
    from data.data import AudioDataLoader

    # DATA PART
    train_json = "data/data.json"
    batch_size = 2
    max_length_in = 1000
    max_length_out = 1000
    num_batches = 10
    num_workers = 2

    with open(train_json, 'rb') as f:
        train_json = json.load(f)['utts']

    train_dataset = AudioDataset(
        train_json, batch_size, max_length_in, max_length_out, num_batches)
    # NOTE: must set batch_size=1 here.
    train_loader = AudioDataLoader(
        train_dataset, batch_size=1, num_workers=num_workers)

    # MODEL PART
    input_size = 83
    hidden_size = 2
    num_layers = 2
    bidirectional = True
    rnn_type = 'lstm'

    encoder = Encoder(input_size, hidden_size, num_layers,
                      bidirectional=bidirectional,
                      rnn_type=rnn_type)
    encoder.cuda()
    for i, (data) in enumerate(train_loader):
        padded_input, input_lengths, targets = data
        padded_input = padded_input.cuda()
        input_lengths = input_lengths.cuda()
        print(i)
        print(padded_input.size())
        print(input_lengths.size())
        output, hidden = encoder(padded_input, input_lengths)
        print(output)
        print(output.size())
        print("*"*20)
