import os
import pickle

from madmom.features import DBNBeatTrackingProcessor
import torch

from model import BeatTrackingNet, TCNNet
from Step1_dataprocess import craet_madmom_feature
import yaml
import sys
import pdb

# import config
with open('/opt/challenge.ai.mgtv/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def load_checkpoint(model, checkpoint_file):
    """
    Restores a model to a given checkpoint, but loads directly to CPU, allowing
    model to be run on non-CUDA devices.
    """
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device('cpu')))


# Some important constants that don't need to be command line params
FFT_SIZE = config['fft_size']
HOP_LENGTH_IN_SECONDS = config['hop_length']
HOP_LENGTH_IN_SAMPLES = 220
N_MELS = config['n_mels']
TRIM_SIZE = (config['trim_size'][0], config['trim_size'][1])
SR = config['sample_rate']

# Paths to checkpoints distributed with the beat tracker. It's possible to
# call the below functions with custom checkpoints also.
DEFAULT_CHECKPOINT_PATH = config['default_checkpoint_path']

# Prepare the models
# model = BeatTrackingNet()
model = TCNNet(input_size=266, output_size=2, nhid=100, levels=6, kernel_size=5, dropout=0.25)
model.eval()

#  Prepare the post-processing dynamic Bayesian networks, courtesy of madmom.
dbn = DBNBeatTrackingProcessor(
    min_bpm=55,
    max_bpm=215,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)


def beat_activations_from_spectrogram(
        spectrogram,
        checkpoint_file=None):
    """
    Given a spectrogram, use the TCN model to compute a beat activation
    function.
    """
    load_checkpoint(model, checkpoint_file)

    # Speed up computation by skipping torch's autograd
    with torch.no_grad():
        # Convert to torch tensor if necessary
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram) \
                .unsqueeze(0) \
                .unsqueeze(0) \
                .float()
        else:
            # Otherwise use the spectrogram as-is
            spectrogram_tensor = spectrogram
        return model(spectrogram_tensor).numpy()


def predict_beats_from_spectrogram(
        spectrogram,
        checkpoint_file=None):
    """
    Given a spectrogram, predict a list of beat times using the TCN model and
    a DBN post-processor.
    """
    raw_activations = \
        beat_activations_from_spectrogram(
            spectrogram,
            checkpoint_file).squeeze()

    beat_activations = raw_activations
    dbn.reset()
    predicted_beats = dbn.process_offline(beat_activations.squeeze())
    return predicted_beats


def beatTracker(input_file, checkpoint_file=None):
    """
    Our main entry point — load an audio file, create a spectrogram and predict
    a list of beat times from it.
    """
    # mag_spectrogram = init_single_spec(
    #     input_file,
    #     FFT_SIZE,
    #     HOP_LENGTH_IN_SECONDS,
    #     N_MELS).T
    # print(mag_spectrogram.shape)
    mag_spectrogram = craet_madmom_feature(input_file).T
    # print(mag_spectrogram.shape)
    return predict_beats_from_spectrogram(
        mag_spectrogram,
        checkpoint_file)

# 将数据写入文件
def write_data(file, data):
    fp = open(file, 'w')
    for i in data:
        i = '{:.2f}'.format(i)
        fp.write(str(i) + "\n")
    fp.close()
    # print(file)

if __name__ == '__main__':
    DATASET_PATH = '/dataset/'
    OUTPUT_PATH = '/output/'
    wav_list = os.listdir(DATASET_PATH)
    checkpoint_file = '/opt/challenge.ai.mgtv/model/_Epoch20.pt'
    # assert wav_list == 50, 'Num Error'
    for wav_file in wav_list:
        if wav_file[-4:] == '.wav':
            wav_dir = os.path.join(DATASET_PATH, wav_file)
            beats = beatTracker(wav_dir, checkpoint_file)
            out_beats = os.path.join(OUTPUT_PATH, wav_file[:-4] + '.beat')
            write_data(out_beats, beats)
            downbeats = []
            out_downbeats = os.path.join(OUTPUT_PATH, wav_file[:-4] + '.downbeat')
            write_data(out_downbeats, downbeats)