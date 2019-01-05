#!/bin/bash

python train.py --train_json ../../test/data/data.json --valid_json ../../test/data/data.json --vocab ../../test/data/train_nodup_sp_units.txt --einput 83 --print_freq 1 --checkpoint --epochs 5 --batch_size 2 &

python recognize.py --recog_json ../../test/data/data.json --vocab ../../test/data/train_nodup_sp_units.txt --result_label ./result.json --model_path exp/temp/final.pth.tar --beam_size 3 --nbest 2 > log &
