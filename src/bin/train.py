import argparse

import torch

from data.data import AudioDataLoader, AudioDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.seq2seq import Seq2Seq
from solver.solver import Solver
from utils.utils import process_dict

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Listen Attend and Spell framework).")
# General config
# Task related
parser.add_argument('--train-json', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid-json', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--vocab', type=str, required=True,
                    help='vocab')
# Network architecture
# encoder
# TODO: automatically infer input dim
parser.add_argument('--einput', default=40, type=int,
                    help='Dim of encoder input')
parser.add_argument('--ehidden', default=512, type=int,
                    help='Size of encoder hidden units')
parser.add_argument('--elayer', default=4, type=int,
                    help='Number of encoder layers.')
parser.add_argument('--ebidirectional', default=True, action='store_true',
                    help='Whether use bidirectional encoder')
parser.add_argument('--etype', default='lstm', type=str,
                    help='Type of encoder RNN')
# attention
parser.add_argument('--atype', default='dot', type=str,
                    help='Type of attention (Only support Dot Product now)')
# decoder
# TODO: automatically infer vocab size/sos id/eos id
# parser.add_argument('--dvocab-size', default=5000, type=int,
#                     help='Size of output vocab')
parser.add_argument('--dembed', default=512, type=int,
                    help='Size of decoder embedding')
# parser.add_argument('--dsos-id', default=0, type=int,
#                     help='End-Of-Sentence index')
# parser.add_argument('--deos-id', default=1, type=int,
#                     help='End-Of-Sentence index')
parser.add_argument('--dhidden', default=512*2, type=int,
                    help='Size of decoder hidden units. Should be encoder '
                    '(2*) hidden size dependding on bidirection')
parser.add_argument('--dlayer', default=1, type=int,
                    help='Number of decoder layers.')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half-lr', dest='half_lr', action='store_true',
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    help='Early stop training when halving lr but still get'
                    'small improvement')
parser.add_argument('--max-norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--batch-size', '-b', default=32, type=int,
                    help='Batch size')
parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='sgd', type=str,
                    choices=['sgd'],
                    help='Optimizer (Only support SGD now)')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
# save and load model
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print-freq', default=10, type=int,
                    help='Frequency of printing training infomation')


def main(args):
    # Construct Solver
    # data
    tr_dataset = AudioDataset(args.train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out)
    cv_dataset = AudioDataset(args.valid_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers)
    # load vocab and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.vocab)
    vocab_size = len(char_list)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    encoder = Encoder(args.einput, args.ehidden, args.elayer,
                      bidirectional=args.ebidirectional, rnn_type=args.etype)
    decoder = Decoder(vocab_size, args.dembed, sos_id,
                      eos_id, args.dhidden, args.dlayer,
                      bidirectional_encoder=args.ebidirectional)
    model = Seq2Seq(encoder, decoder)
    print(model)
    model.cuda()
    # optimizer
    optimizier = torch.optim.SGD(model.parameters(),
                                 lr=args.lr,
                                 momentum=args.momentum)
    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
