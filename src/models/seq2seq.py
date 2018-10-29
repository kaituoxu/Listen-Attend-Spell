import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        loss = self.decoder(padded_target, encoder_padded_outputs)
        return loss

    @classmethod
    def load_model(cls, path):
        package = torch.load(path)
        model = self.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        encoder = Encoder(package['einput'],
                          package['ehidden'],
                          package['elayer'],
                          bidirectional=package['ebidirectional'],
                          rnn_type=package['etype'])
        decoder = Decoder(package['dvocab_size'],
                          package['dembed'],
                          package['dsos_id'],
                          package['deos_id'],
                          package['dhidden'],
                          package['dlayer'],
                          bidirectional_encoder=package['ebidirectional']
                          )
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch):
        package = {
            # encoder
            'einput': model.encoder.input_size,
            'ehidden': model.encoder.hidden_size,
            'elayer': model.encoder.num_layers,
            'ebidirectional': model.encoder.bidirectional,
            'etype': model.encoder.rnn_type,
            # decoder
            'dvocab_size': model.decoder.vocab_size,
            'dembed': model.decoder.embedding_dim,
            'dsos_id': model.decoder.sos_id,
            'deos_id': model.decoder.eos_id,
            'dhidden': model.decoder.hidden_size,
            'dlayer': model.decoder.num_layers,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        return package
