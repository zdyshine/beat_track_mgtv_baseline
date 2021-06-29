import os
import librosa
import numpy as np
import yaml
import sys

def AdjustLearningRate(optimizer, lr):
    for param_group in optimizer.param_groups:
        print('param_group',param_group['lr'])
        param_group['lr'] = lr

# import config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class SingleFile(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path


def init_all_specs(
        input_folder=config['dataset_folder_train'],
        output_folder=config['spec_folder_train'],
        n_fft=config['fft_size'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']):
    """Preprocess spectrograms for all data.

    Args:
        input_folder: raw data path.
        output_folder: spectrogram path
        n_fft: n_fft
        hop_length: hop_length per second
        n_mels: n_mels

    Returns:

    """
    # scan folder and find all *.wav files
    file_list = list()
    for folder in os.listdir(input_folder)[:1]:
        out_folder = '/'.join([output_folder, folder])
        folder_path = '/'.join([input_folder, folder])
        os.makedirs(out_folder, exist_ok=True)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    # print(file)
                    file_path = '/'.join([input_folder, folder, file])
                    file_name = file[:-4]  # cut-off '.wav'
                    file_list.append(SingleFile(file_name, file_path))

                    # create spec for each file
                    spec = init_single_spec(file_path, n_fft, hop_length, n_mels) # 81, 3009
                    # spec = init_madmom_spec(file_path, n_fft, hop_length, n_mels)
                    print(spec.shape)


def init_single_spec(
        file_path=None,
        n_fft=config['fft_size'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']):
    """ Create a spectrogram for single data.

    Args:
        file_path: the file path of single raw data
        n_fft: n_fft
        hop_length: hop length per second
        n_mels: n_mels

    Returns:
        numpy spectrogram

    """
    x, sr = librosa.load(file_path)
    # print(sr) # 22050
    hop_length_in_samples = int(np.floor(hop_length * sr))
    # print(hop_length_in_samples) # 220
    spec = librosa.feature.melspectrogram(
        x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels)
    return np.abs(spec)


def precision(prediction, target, tolerance=0.07):
    """
    Calculates the precision of a prediction, given the target.

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Keyword Arguments:
        tolerance {float} -- Tolerance in seconds (default: {0.07})

    Returns:
        float -- Precision measure for given prediction and target vectors.
    """
    num_correct = 0.0
    pred_beats = prediction.tolist().copy()
    for true_beat in target:
        for predicted_beat in pred_beats:
            if abs(true_beat - predicted_beat) <= tolerance:
                num_correct += 1.0
                pred_beats.remove(predicted_beat)
                break

    return num_correct / len(prediction)


def recall(prediction, target, tolerance=0.07):
    """
    Calculates the recall of a prediction, given the target.

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Keyword Arguments:
        tolerance {float} -- Tolerance in seconds (default: {0.07})

    Returns:
        float -- Recall measure for given prediction and target vectors.
    """
    num_correct = 0.0
    false_negatives = 0.0
    pred_beats = prediction.tolist().copy()

    for true_beat in target:
        false_negatives += 1.0
        for predicted_beat in pred_beats:
            if abs(true_beat - predicted_beat) <= tolerance:
                num_correct += 1.0
                false_negatives -= 1.0
                pred_beats.remove(predicted_beat)
                break

    return num_correct / (num_correct + false_negatives)


def f_measure(prediction, target, tolerance=0.07):
    """
    Calculates the f-measure of a prediction, given the target

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Keyword Arguments:
        tolerance {float} -- Tolerance in seconds (default: {0.07})

    Returns:
        float -- f-measure for given prediction and target vectors.
    """
    r = recall(prediction, target, tolerance)
    p = precision(prediction, target, tolerance)
    return 2 * r * p / max(r + p, sys.float_info.epsilon)


def nearest_value(array, value):
    """
    Searches array for the closest value to a given target.

    Arguments:
        array {NumPy Array} -- A NumPy array of numbers.
        value {float/int} -- The target value.

    Returns:
        float/int -- The closest value to the target value found in the array.
    """
    return array[np.abs(array - value).argmin()]


def cemgil_accuracy(prediction, target):
    """
    Calculates the accuracy score proposed in Cemgil et al 2001 [2], using
    a Gaussian error function.

    Arguments:
        prediction {list/NumPy Array} -- List of timings of predicted beats, in
                                         seconds.
        target {list/NumPy Array} -- List of timings of annotated beats, in
                                     seconds.

    Returns:
        float -- Cemgill Accuracy score for given prediction and target vectors
    """
    def w(x):
        variance = 0.04
        return np.exp(-(x**2) / (2 * variance ** 2))

    B = prediction.shape[0]
    J = target.shape[0]
    sigma = 0.0
    for a in target:
        gamma = nearest_value(prediction, a)
        sigma += w(gamma - a)

    return sigma / ((B + J) * 0.5)
