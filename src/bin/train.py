#!/usr/bin/env python
import argparse

import torch

from data import AudioDataLoader, AudioDataset
from decoder import Decoder
from encoder import Encoder
from seq2seq import Seq2Seq
from solver import Solver
from utils import process_dict

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Listen Attend and Spell framework).")
# General config
# Task related
parser.add_argument('--train-json', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid-json', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
# Network architecture
# encoder
# TODO: automatically infer input dim
parser.add_argument('--einput', default=80, type=int,
                    help='Dim of encoder input')
parser.add_argument('--ehidden', default=512, type=int,
                    help='Size of encoder hidden units')
parser.add_argument('--elayer', default=4, type=int,
                    help='Number of encoder layers.')
parser.add_argument('--edropout', default=0.0, type=float,
                    help='Encoder dropout rate')
parser.add_argument('--ebidirectional', default=1, type=int,
                    help='Whether use bidirectional encoder')
parser.add_argument('--etype', default='lstm', type=str,
                    help='Type of encoder RNN')
# attention
parser.add_argument('--atype', default='dot', type=str,
                    help='Type of attention (Only support Dot Product now)')
# decoder
parser.add_argument('--dembed', default=512, type=int,
                    help='Size of decoder embedding')
parser.add_argument('--dhidden', default=512*2, type=int,
                    help='Size of decoder hidden units. Should be encoder '
                    '(2*) hidden size dependding on bidirection')
parser.add_argument('--dlayer', default=1, type=int,
                    help='Number of decoder layers.')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half-lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early-stop', dest='early_stop', default=0, type=int,
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
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print-freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom-id', default='LAS training',
                    help='Identifier for visdom run')


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
    # load dictionary and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    encoder = Encoder(args.einput, args.ehidden, args.elayer,
                      dropout=args.edropout, bidirectional=args.ebidirectional,
                      rnn_type=args.etype)
    decoder = Decoder(vocab_size, args.dembed, sos_id,
                      eos_id, args.dhidden, args.dlayer,
                      bidirectional_encoder=args.ebidirectional)
    model = Seq2Seq(encoder, decoder)
    print(model)
    model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
