import os

import librosa
import numpy as np
import yaml
from tqdm import tqdm

def create_spectrogram(
        file_path,
        n_fft,
        hop_length,
        n_mels):
    x, sr = librosa.load(file_path)
    hop_length_in_samples = int(np.floor(hop_length * sr))
    spec = librosa.feature.melspectrogram(
        x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels)
    mag_spec = np.abs(spec)

    return mag_spec

def create_spectrograms(
        audio_dir,
        spectrogram_dir,
        n_fft,
        hop_length,
        n_mels):
    # print(os.listdir(audio_dir))
    for folder in os.listdir(audio_dir):
        print('=========> Processing {}.'.format(folder))
        out_folder = '/'.join([spectrogram_dir, folder])
        folder_path = '/'.join([audio_dir, folder])
        os.makedirs(out_folder, exist_ok=True)
        if os.path.isdir(folder_path):
            for file in tqdm(os.listdir(folder_path)):
                if file.endswith('.wav'):
                    # print(file)
                    file_path = '/'.join([audio_dir, folder, file])
                    file_name = file[:-4]  # cut-off '.wav'

                    # create spec for each file
                    spec = create_spectrogram(file_path, n_fft, hop_length, n_mels)
                    np.save('/'.join([out_folder, file_name]), spec)


def trim_spectrogram(spectrogram, trim_size):
    output = np.zeros(trim_size)
    dim0_range = min(trim_size[0], spectrogram.shape[0])
    dim1_range = min(trim_size[1], spectrogram.shape[1])

    output[:dim0_range, :dim1_range] = spectrogram[:dim0_range, :dim1_range]
    return output


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('=========> process train data')
    create_spectrograms(
        audio_dir=config['dataset_folder_train'],
        spectrogram_dir=config['spec_folder_train'],
        n_fft=config['fft_size'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']
    )

    print('=========> process validation data')
    create_spectrograms(
        audio_dir=config['dataset_folder_val'],
        spectrogram_dir=config['spec_folder_val'],
        n_fft=config['fft_size'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']
    )
