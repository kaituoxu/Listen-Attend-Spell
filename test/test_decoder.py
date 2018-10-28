import torch

from models.decoder import Decoder
from utils.utils import IGNORE_ID

if __name__ == "__main__":
    VOC, EMB, SOS, EOS, H, L = 10, 20, 8, 9, 2, 2
    N, To, Ti = 4, 5, 3
    decoder = Decoder(10, 20, 8, 9, 2, 2)
    print(decoder)
    padded_input = torch.randint(10, (N, To), dtype=torch.long)  # N x To
    padded_input[-1, -3:] = IGNORE_ID
    encoder_padded_outputs = torch.randn(N, Ti, H)  # N x Ti x H
    print(padded_input)
    print(padded_input.size())
    print(encoder_padded_outputs)
    print(encoder_padded_outputs.size())
    decoder_outputs, decoder_hidden = decoder(
        padded_input, encoder_padded_outputs)
    print(decoder_outputs)
    print(len(decoder_outputs))  # To + 1?
    print(decoder_outputs[0].size())  # N x VOC
