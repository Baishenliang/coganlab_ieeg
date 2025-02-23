from typing import List, Tuple
from os import PathLike
from ieeg.decoding.decoders import PcaLdaClassification
from ieeg.calc.mat import LabeledArray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ieeg.calc.oversample import MinimumNaNSplit
from ieeg.calc.fast import mixup
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as p
from stm import io
from stm.stats import augment_data
import argparse
import pickle
class Decoder(PcaLdaClassification, MinimumNaNSplit):
    def __init__(self, categories: dict, *args,
                 n_splits: int = 5,
                 n_repeats: int = 5,
                 oversample: bool = True,
                 max_features: int = float(“inf”),
                 **kwargs):
        PcaLdaClassification.__init__(self, *args, **kwargs)
        MinimumNaNSplit.__init__(self, n_splits, n_repeats, min_non_nan=-1)
        if not oversample:
            self.oversample = lambda x, func, axis: x
        self.categories = categories
        self.max_features = max_features
    def cv_cm(self, x_data: np.ndarray, labels: np.ndarray,
              normalize: str = None, obs_axs: int = -2, augment=None):
        n_cats = len(set(labels))
        mats = np.zeros((self.n_repeats, self.n_splits, n_cats, n_cats))
        obs_axs = x_data.ndim + obs_axs if obs_axs < 0 else obs_axs
        idx = [slice(None) for _ in range(x_data.ndim)]
        for f, (train_idx, test_idx) in enumerate(self.split(x_data.swapaxes(
                0, obs_axs), labels)):
            x_train = np.take(x_data, train_idx, obs_axs)
            x_test = np.take(x_data, test_idx, obs_axs)
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            # Augment data after split to avoid data leakage
            if augment != None:
                x_train, y_train = augment_data(x_train, y_train, method=augment)
            for i in set(labels):
                # fill in train data nans with random combinations of
                # existing train data trials (mixup)
                idx[obs_axs] = y_train == i
                x_train[tuple(idx)] = self.oversample(x_train[tuple(idx)],
                                                      axis=obs_axs,
                                                      func=mixup)
            # fill in test data nans with noise from distribution
            is_nan = np.isnan(x_test)
            x_test[is_nan] = np.random.normal(0, 1, np.sum(is_nan))
            # feature selection
            train_in = flatten_features(x_train, obs_axs)
            test_in = flatten_features(x_test, obs_axs)
            if train_in.shape[1] > self.max_features:
                tidx = np.random.choice(train_in.shape[1], self.max_features,
                                        replace=False)
                train_in = train_in[:, tidx]
                test_in = test_in[:, tidx]
            # fit model and score results
            self.fit(train_in, y_train)
            pred = self.predict(test_in)
            rep, fold = divmod(f, self.n_splits)
            mats[rep, fold] = confusion_matrix(y_test, pred)
        # average the repetitions, sum the folds
        matk = np.sum(mats, axis=1)
        if normalize == ‘true’:
            divisor = np.sum(matk, axis=-1, keepdims=True)
        elif normalize == ‘pred’:
            divisor = np.sum(matk, axis=-2, keepdims=True)
        elif normalize == ‘all’:
            divisor = self.n_repeats
        else:
            divisor = 1
        return matk / divisor
def flatten_features(arr: np.ndarray, obs_axs: int = -2) -> np.ndarray:
    obs_axs = arr.ndim + obs_axs if obs_axs < 0 else obs_axs
    if obs_axs != 0:
        out = arr.swapaxes(0, obs_axs)
    else:
        out = arr.copy()
    return out.reshape(out.shape[0], -1)
def find_min_labels(array_dict):
    ‘’'Takes the dictionary of labeled arrays across subjects and finds the minimum number
    across the arrays for each label.‘’'
    count_arr = np.zeros(1)
    for _, arr in array_dict.items():
        lbls, counts = np.unique(arr.labels[0], return_counts=True)
        if np.all(count_arr == 0):
            count_arr = counts
        else:
            count_arr = np.vstack((count_arr, counts))
    minimum_counts = np.min(count_arr, axis=0)
    label_mins = dict(zip(lbls, minimum_counts))
    return label_mins
def make_cross_pt_array(array_dict):
    ‘’'Makes array across subjects in input array_dict for use with
    cross-patient decoding’‘'
    label_mins = find_min_labels(array_dict)
    ch_labels = []
    run = 0
    for _, arr in array_dict.items():
        run += 1
        take_idx = []
        label_counts = dict(zip(arr.labels[0], np.zeros(len(arr.labels[0]))), dtype=int)
        for idx, lbl in enumerate(arr.labels[0]):
            label_counts[lbl] += 1
            if label_counts[lbl] <= label_mins[lbl]:
                take_idx.append(idx)
        new_arr = arr.take(take_idx, axis=0)
        sorted_idx = np.argsort(new_arr.labels[0])
        sorted_new_arr = new_arr[sorted_idx, :, :]
        labels = sorted_new_arr.labels[0]
        time_labels = sorted_new_arr.labels[2]
        ch_labels.extend(sorted_new_arr.labels[1])
        if run == 1:
            all_pt_arr = new_arr
        else:
            all_pt_arr = np.concatenate((all_pt_arr, new_arr), axis=1)
    final_arr = LabeledArray(all_pt_arr)
    final_arr.labels = [labels, ch_labels, time_labels]
    return final_arr
def run_decoding(subjects: List, LAB_root: PathLike, stats_folder: PathLike, epoch: str,
    classify: str, out_file: PathLike=None, augment: str=None, times: Tuple=None,
    correct_only: bool=False):
    ‘’'Creates a labeled array from input subjects and then performs
    single subject decoding if input is a string or cross-patient decoding
    if input is a list
    ------------------------
    Parameters:
    subjects: list containing integers of subject numbers
    LAB_root: str or PathLike to lab data folder
    stats_folder: str or PathLike to folder containing significnat channels for task
    epoch: which epoch to use for data: ‘Auditory’, ‘All stims’, ‘Delay’, ‘Probe’, and ‘Response’
    are current options
    classify: string corresponding to which classification task to run: current options below
        ‘word_or_no’: word/non-word decoding
        ‘load’: stimulus load decoding (i.e. 3, 5, 7, 9)
        ‘load_binary’: stimulus load decoding (3, 5 vs. 7, 9)
        ‘initial_phoneme’: decode initial phoneme of 7 most common phonemes
        ‘token’: decode token identify (40 different tokens in trials)
        ‘correct_or_no’: decode correct vs incorrect trial
        ‘match_mismatch’: decode whether probe was in or out of list
    out_file: str or PathLike for output figure if desired
    augment: str, corresponding to augmentation strategy in stm.stats.data_augment
    times: tuple, corresponding to t min and tmax to use around given epoch to make
    trials
    correct_only: bool, if
    ‘’'
    assert type(subjects) is list, ‘subjects variable must be list’
    array_dict = {}
    for subj in subjects:
        # Get trial epochs and sternberg trial info
        tri, st_tri = io.get_trials(subj, LAB_root, stats_folder, epoch,
        extract=‘gamma’, correct_only=correct_only, input_times=times)
        array_dict[subj] = io.make_classifier_array(tri, st_tri, classify)
        if len(subjects) == 1:
            final_arr = array_dict[subj]
    if len(subjects) > 1:
        final_arr = make_cross_pt_array(array_dict)
    class_ids = final_arr.labels[0]
    classes = {k: i for i, k in enumerate(np.unique(class_ids))}
    labels = np.array([classes[k] for k in class_ids])
    decoder = Decoder(classes, 0.80, oversample=True, n_splits=5, n_repeats=10)
    cm = decoder.cv_cm(final_arr.__array__().swapaxes(0, 1), labels, normalize=‘true’, augment=augment)
    cm = np.mean(cm, axis=0)
    if out_file is not None:
        _, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes.keys())
        disp.plot(ax=ax, text_kw={‘fontsize’: 6})
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.savefig(out_file)
        with open(p(out_file).with_suffix(‘.pkl’), ‘wb’) as f:
            pickle.dump(cm, f, pickle.HIGHEST_PROTOCOL)
    return cm, final_arr
if __name__ == ‘__main__‘:
    parser = argparse.ArgumentParser(description=‘perform delay decoding’)
    parser.add_argument(‘-s’, ‘--subjects’, nargs=‘*’, help=‘subject number’, required=True)
    parser.add_argument(‘LAB_root’, help=‘lab root data folder’)
    parser.add_argument(‘stats_folder’, help=‘folder containing stats .csv files’)
    parser.add_argument(‘-e’, ‘--epoch’, help=‘Epochs to run decoding on’)
    parser.add_argument(‘-c’, ‘--classify’, help=‘Classify task to perform’)
    parser.add_argument(‘-of’, ‘--out_file’, help=‘file to save figures’)
    parser.add_argument(‘-a’, ‘--augment’, help=‘augmentation method to use’)
    parser.add_argument(‘-t’, ‘--times’, nargs=‘+’, help = ‘times to use around epoch’,
                        type=float)
    parser.add_argument(‘--correct_only’, ‘-co’, help=‘calculate stats only for trials with correct response’, action=‘store_true’)
    args = parser.parse_args()
    run_decoding(args.subjects, args.LAB_root, args.stats_folder, args.epoch, \
         args.classify, args.out_file, args.augment, tuple(args.times), args.correct_only)