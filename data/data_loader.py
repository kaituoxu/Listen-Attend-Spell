from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import scipy.io.wavfile as wavfile

import signal_process


#
# Audio Parser
#

class AudioParser(object):
    """
    Audio processing API.

    We can design FilterBankAudioParser, MFCCAudioParser, etc.
    """
    def parse_audio(self, audio_path):
        raise NotImplementedError


class FilterBankAudioParser(AudioParser):
    """
    Audio filter bank feature extraction.
    Based on Kaldi fbank computation.
    """
    def __init__(self, audio_conf):
        super(FilterBankAudioParser, self).__init__()
        # Frame extraction options
        # TODO: I set a lot of default option here, make them option later
        self.samp_freq = audio_conf["samp_freq"]
        self.frame_length_ms = 25.0
        self.frame_shift_ms = 10.0
        self.dither = 0.0
        self.preemph_coeff = 0.97
        self.remove_dc_offset = True
        self.window_type = "hamming" # hamming or povey
        self.round_to_power_of_two = True
        self.num_mel_bins = 40
        self.low_freq = 20
        self.high_freq = None

    def parse_audio(self, audio_path):
        """
        Extract filter bank feature from raw audio in Kaldi way.
        """
        samplerate, signal = wavfile.read(audio_path)
        assert(self.samp_freq == samplerate)
        fbank_feat = signal_process.kaldi_logfbank(
            signal,
            samplerate=samplerate,
            winlen=self.frame_length_ms / 1000.0,
            winstep=self.frame_shift_ms / 1000.0,
            nfilt=self.num_mel_bins,
            round_to_power_of_two=self.round_to_power_of_two,
            lowfreq=self.low_freq,
            highfreq=self.high_freq,
            dither=self.dither,
            remove_dc_offset=self.remove_dc_offset,
            preemph=self.preemph_coeff,
            wintype=self.window_type)
        return fbank_feat


#
# Transcript Parser
#

class TranscriptParser(object):
    """
    Transcript Processing API.

    We can design CharTranscriptParser, WordTranscriptParser, etc.
    """
    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class CharTranscriptParser(TranscriptParser):
    """
    Using Chinese char as transcript unit.
    """
    def __init__(self, char_list):
        """
        Arguments:
            char_list: each element is a Chinese char.
        """
        super(CharTranscriptParser, self).__init__()
        self.char_list = char_list
        self.char2id = dict([(char_list[i], i) for i in range(len(char_list))])

    def parse_transcript(self, transcript_path):
        """
        Read the transcript and convert it to id list.

        One transcript is a line of char sequence splited by ' ':
            "char1 char2 char3 char4"
        """
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        # list() will transform one string to a char list.
        transcript = [self.char2id.get(x) for x in list(transcript)]
        return transcript


#
# Audio Dataset (Audio + Transcript)
#

class AudioDataset(Dataset):
    def __init__(self, audio_parser, trans_parser, audio_trans_csv_file):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        Based on AudioParser and TranscriptParser API, we can use different
        AudioParser and TranscriptParser.

        Arguments:
        """
        super(AudioDataset, self).__init__()
        self.audio_parser = audio_parser
        self.trans_parser = trans_parser
        with open(audio_trans_csv_file) as f:
            audio_trans_paths = f.readlines()
        audio_trans_paths = [path.strip().split(',') for path in audio_trans_paths]
        self.audio_trans_paths = audio_trans_paths
        self.num_utterances = len(paths)

    def __getitem__(self, index):
        audio_trans_path = self.audio_trans_paths[index]
        audio_path, trans_path = audio_trans_path[0], audio_trans_path[1]
        features = self.audio_parser.parse_audio(audio_path)
        labels = self.trans_parser.parse_transcript(trans_path)
        return features, labels

    def __len__(self):
        return self.num_utterances


#
# AudioDataLoader
#

def _collate_fn(batch):
    pass


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Create a data loader for AudioDataset.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly
        sized samples together.
        (That means you should put the similarly sized samples together out of
         this code. See an4.py as an example.)
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        pass

    def __len__(self):
        pass
