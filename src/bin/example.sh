#!/bin/bash

python train.py --train-json ../../test/data/data.json --valid-json ../../test/data/data.json --vocab ../../test/data/train_nodup_sp_units.txt --einput 83 --print-freq 1 --checkpoint --epochs 5 --batch-size 2 &

python recognize.py --recog-json ../../test/data/data.json --vocab ../../test/data/train_nodup_sp_units.txt --result-label ./result.json --model-path exp/temp/final.pth.tar --beam-size 3 --nbest 2 > log &
