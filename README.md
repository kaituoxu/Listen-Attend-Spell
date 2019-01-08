# Listen, Attend and Spell
A PyTorch implementation of Listen, Attend and Spell (LAS) [1], an end-to-end automatic speech recognition framework, which directly converts acoustic features to character sequence using only one nueral network.

## Install
- Python3 (Recommend Anaconda)
- PyTorch 0.4.1+
- [Kaldi](https://github.com/kaldi-asr/kaldi) (Just for feature extraction)
- `pip install -r requirements.txt`
- `cd tools; make KALDI=/path/to/kaldi`
- If you want to run `egs/aishell/run.sh`, download [aishell](http://www.openslr.org/33/) dataset for free.

## Usage
1. `$ cd egs/aishell` and modify aishell data path to your path in `run.sh`.
2. `$ bash run.sh`, that's all!

You can change hyper-parameter by `$ bash run.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.
### More detail
```bash
$ cd egs/aishell/
$ . ./path.sh
```
Train
```bash
$ train.py -h
```
Decode
```bash
$ recognize.py -h
```
### Workflow
Workflow of `egs/aishell/run.sh`:
- Stage 0: Data Preparation
- Stage 1: Feature Generation
- Stage 2: Dictionary and Json Data Preparation
- Stage 3: Network Training
- Stage 4: Decoding
### Visualize loss
If you want to visualize your loss, you can use `visdom` to do that:
- Open a new terminal in your remote server (recommend tmux) and run `$ visdom`.
- Open a new terminal and run `$ bash run.sh --visdom 1 --visdom_id "<any-string>"` or `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`.
- Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`.
- In visdom website, chose `<any-string>` in `Environment` to see your loss.

## Results
| Model | CER | Config |
| :---: | :-: | :----: |
| LSTMP | 9.85| 4x(1024-512) |
| Listen, Attend and Spell | 13.2 | See egs/aishell/run.sh |

## Reference
[1] W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in ICASSP 2016. (https://arxiv.org/abs/1508.01211v2)
