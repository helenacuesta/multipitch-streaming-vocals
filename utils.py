# util functions
import sys
import os
import json
import csv

import numpy as np
import pandas as pd

import pumpp
import jams
import librosa
import mir_eval
import muda

from scipy.ndimage import filters
import scipy

import config

import tensorflow.keras.backend as K
import tensorflow as tf

'''
General util functions
Some of the functions in this file are taken/adapted from deepsalience.
'''

RANDOM_STATE = 42

def compute_hcqt(audiofile):

    cqtm = []

    y, _ = librosa.load(audiofile, sr=config.fs)

    for h in config.harmonics:
        C = librosa.cqt(y=y, sr=config.fs, hop_length=config.hopsize,
                fmin=config.f_min * h,
                n_bins=(config.n_octaves * config.over_sample * 12),
                bins_per_octave=(config.over_sample * 12))

        C, P = librosa.magphase(C)

        C = librosa.amplitude_to_db(C)
        C = (C - C.min()) / (C.max() - C.min())
        cqtm.append(C)

    cqtm = np.asarray(cqtm).astype(np.float32)
    cqtm = np.moveaxis(cqtm, 0, -1)

    return cqtm

def shift_annotations(jams_path, jams_fname, audio_path, audio_fname):

    '''Use the IRConvolution deformer to shift F0 annotations according to
    the estimated group delay introduced by impulse response
    '''

    ir_muda = muda.deformers.IRConvolution(ir_files='./ir/IR_greathall.wav', n_fft=2048, rolloff_value=-24)

    # make sure the duration field in the jams file is not null
    jm = jams.load(os.path.join(jams_path, jams_fname))
    jm.annotations[0].duration = jm.file_metadata.duration
    jm.save(os.path.join(jams_path, jams_fname))

    # load jam and associated audio
    jam = muda.load_jam_audio(os.path.join(jams_path, jams_fname), os.path.join(audio_path, audio_fname))


    for s in ir_muda.states(jam):
        ir_muda.deform_times(jam.annotations[0], s)

    # store deformed annotations in the reverb folder
    jam.save(os.path.join(jams_path, 'rev_' + jams_fname))


def save_json_data(data, save_path):
    with open(save_path, 'w') as fp:
        json.dump(data, fp)


def load_json_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data


def get_stft_params():

    bins_per_octave = config.bins_per_octave
    n_octaves = config.n_octaves
    over_sample = config.over_sample
    harmonics = config.harmonics
    sr = config.fs
    fmin = config.f_min
    hop_length = config.hopsize

    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length, over_sample

def bkld(y_true, y_pred):
    """Brian's KL Divergence implementation (cross-entropy)
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    # return K.mean(K.mean(-1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),axis=-1), axis=-1)
    return K.mean(-1.0 * y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred), axis=-1)


def soft_binary_accuracy(y_true, y_pred):
    """Binary accuracy that works when inputs are probabilities
    """
    # return K.mean(K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1), axis=-1)
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)

def pyin_to_unvoiced(pyin_path, pyin_fname, audio_path, audio_fname, fs=config.fs):

    '''This function takes a CSV file with smoothedpitchtrack info from pYIN
    and adds zeros in the unvoiced frames.
    '''

    x, fs = librosa.core.load(os.path.join(audio_path, audio_fname), sr=fs)

    if pyin_fname.endswith('csv'):
        pyi = pd.read_csv(os.path.join(pyin_path, pyin_fname), header=None).values

    elif pyin_fname.endswith('f0'):
        pyi = np.loadtxt(os.path.join(pyin_path, pyin_fname))

    else:
        print("Wrong annotation file format found.")
        quit()


    l_samples = len(x)
    del x
    time_pyin = mir_eval.melody.constant_hop_timebase(hop=config.hopsize, end_time=l_samples) / fs

    # times_pyin uses the same hopsize as the original pyin so we can directly compare them
    pyin_new = np.zeros([len(time_pyin), 2])
    _, _, idx_y = np.intersect1d(np.around(pyi[:, 0], decimals=5), np.around(time_pyin, decimals=5), return_indices=True)
    pyin_new[idx_y, 1] = pyi[:, 1]
    pyin_new[:, 0] = time_pyin

    pd.DataFrame(pyin_new).to_csv(os.path.join(pyin_path, 'constant_timebase', pyin_fname), header=None, index=False)


def get_freq_grid():
    """Get the stft frequency grid
    """
    # freq_grid = librosa.fft_frequencies(sr=config.fs, n_fft=config.stft_size)
    freq_grid = librosa.cqt_frequencies(n_bins=config.num_features, fmin=config.f_min, bins_per_octave=config.bins_per_octave)

    return freq_grid


def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    """
    hop_length = config.hopsize
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=config.fs, hop_length=hop_length
    )
    return time_grid


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def compute_targets(annot_path, annot_fname, stft):


    d = jams.load(os.path.join(annot_path, annot_fname))
    data = np.array(d.annotations[0].data)[:, [0, 2]]

    times = data[:, 0]
    freqs = []
    for d in data[:, 1]:
        freqs.append(d['frequency'])
    freqs = np.array(freqs)

    freq_grid = get_freq_grid()
    time_grid = get_time_grid(n_time_frames=stft.shape[1])

    annot_target = create_annotation_target(
        freq_grid, time_grid, times, freqs)

    np.save(os.path.join(config.feat_targ_folder, 'targets', annot_fname.replace('jams', 'npy')), annot_target)

    return annot_target


def feature_extraction(audio_fname, x, save=True):

    # if x == 0:
    #     x, _ = librosa.load(
    #         os.path.join(config.audio_save_folder, audio_fname),
    #         sr=config.fs
    #     )

    cqt = np.abs(librosa.cqt(
        y=x, sr=config.fs, hop_length=config.hopsize, fmin=config.f_min, n_bins=config.num_features, bins_per_octave=config.bins_per_octave
    ))

    cqt_db = librosa.amplitude_to_db(cqt)
    cqt_db = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min())

    # S = np.abs(
    #     librosa.stft(
    #         y=x, n_fft=config.stft_size, hop_length=config.hopsize))

    if save:
        np.save(os.path.join(config.feat_targ_folder, 'inputs', audio_fname.replace('wav', 'npy')), cqt_db)

    return cqt_db


def save_data(save_path, input_path, output_path, prefix, X, Y, f, t):

    i_path = os.path.join(save_path, 'inputs')
    o_path = os.path.join(save_path, 'outputs')

    if not os.path.exists(i_path):
        os.mkdir(i_path)
    if not os.path.exists(o_path):
        os.mkdir(o_path)

    if not os.path.exists(input_path):

        np.save(input_path, X, allow_pickle=True)
        np.save(output_path, Y, allow_pickle=True)
        print("    Saved inputs and targets targets for {} to {}".format(prefix, save_path))

    else:
        np.save(output_path, Y, allow_pickle=True)
        print("    Saved only targets for {} to {}".format(prefix, save_path))

def get_all_pitch_annotations(mtrack):
    '''Load annotations
    '''

    annot_times = []
    annot_freqs = []

    for stem in mtrack['annot_files']:

        '''Load annotations for each singer in the mixture
        '''
        d = jams.load(os.path.join(mtrack['annot_folder'], stem))
        data = np.array(d.annotations[0].data)[:, [0, 2]]


        times = data[:, 0]
        freqs = []
        for d in data[:, 1]:
            freqs.append(d['frequency'])
        freqs = np.array(freqs)

        '''
        times = data[:, 0]
        freqs = data[:, 1]
        '''

        if data is not None:
            idx_to_use = np.where(freqs > 0)[0]
            times = times[idx_to_use]
            freqs = freqs[idx_to_use]

            annot_times.append(times)
            annot_freqs.append(freqs)
        else:
            print('Data not available for {}.'.format(mtrack))
            continue

    # putting all annotations together
    if len(annot_times) > 0:
        annot_times = np.concatenate(annot_times)
        annot_freqs = np.concatenate(annot_freqs)

        return annot_times, annot_freqs

    else:
        return None, None

def create_annotation_target(freq_grid, time_grid, annotation_times, annotation_freqs):
    """Create the binary annotation target labels with Gaussian blur
    """
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target = np.zeros((n_freqs, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1


    annotation_target_blur = filters.gaussian_filter1d(
        annotation_target, 1, axis=0, mode='constant'
    )
    if len(annot_freq_idx) > 0:
        min_target = np.min(
            annotation_target_blur[annot_freq_idx, annot_time_idx]
        )
    else:
        min_target = 1.0

    annotation_target_blur = annotation_target_blur / min_target
    annotation_target_blur[annotation_target_blur > 1.0] = 1.0

    return annotation_target_blur



def compute_features_mtrack(mtrack, save_dir, wavmixes_path, idx):

    print("Processing {}...".format(mtrack['filename']))

    compute_multif0_complete(mtrack, save_dir, wavmixes_path)

def create_data_split(mtrack_dict, output_path, train_perc=0.75, validation_perc = 0.1):


    """This function creates a fully-randomized
    """

    mtracks = mtrack_dict.keys()

    all_tracks = [
        m for m in mtracks
    ]

    '''
    for m in mtracks:
        if 'reverb' in mtrack_dict[m]['audiopath']:
            all_tracks.append('rev_' + m)
        else:
            all_tracks.append(m)
    '''


    Ntracks = len(all_tracks)

    # randomize track names
    mtracks_randomized = np.random.permutation(all_tracks)

    train_set = mtracks_randomized[:int(train_perc * Ntracks)]
    validation_set = mtracks_randomized[int(train_perc * Ntracks):int(train_perc * Ntracks) + int(validation_perc * Ntracks)]
    test_set = mtracks_randomized[int(train_perc * Ntracks) + int(validation_perc * Ntracks):]

    data_splits = {
        'train': list(train_set),
        'validate': list(validation_set),
        'test': list(test_set)
    }

    with open(output_path, 'w') as fhandle:
        fhandle.write(json.dumps(data_splits, indent=2))

    return train_set, validation_set, test_set



def progress(count, total, suffix=''):
    """
    Function to diplay progress bar
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def get_single_prediction(model, input_fname):

    # we first deal with the scenario of having an audio as input
    if input_fname.endswith('wav'):
        x, _ = librosa.load(
            os.path.join(config.audio_save_folder, input_fname), sr=config.fs
        )

        S = feature_extraction(input_fname, x, save=False)

    # here we deal with the scenario where we have pre-computed features
    elif input_fname.endswith('npy'):
        S = np.load(input_fname)

    else:
        raise ValueError("Please specify either an audio file or a npy file"
                         "with the corresponding spectrogram.")

    L = S.shape[1] - config.patch_len
    idx_slices = np.arange(0, L, config.patch_len)

    output_predictions = {}
    inputs = []

    for t in idx_slices:
        inputs.append(S[:, t:t+config.patch_len][:, :, np.newaxis])

    input_data = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size=32)
    print("\n Prediction starts now...")
    p = model.predict(input_data, verbose=1)
    # import pdb;pdb.set_trace()

    output_predictions["sop"] = np.hstack(p[0])
    output_predictions["alt"] = np.hstack(p[1])
    output_predictions["ten"] = np.hstack(p[2])
    output_predictions["bas"] = np.hstack(p[3])

    return output_predictions

def get_single_prediction_ld(model, input_fname):

    # we first deal with the scenario of having an audio as input
    if input_fname.endswith('wav'):
        x, _ = librosa.load(
            os.path.join(config.audio_save_folder, input_fname), sr=config.fs
        )

        S = feature_extraction(input_fname, x, save=False)

    # here we deal with the scenario where we have pre-computed features
    elif input_fname.endswith('npy'):
        S = np.load(input_fname)

    else:
        raise ValueError("Please specify either an audio file or a npy file"
                         "with the corresponding spectrogram.")

    L = S.shape[1] - config.patch_len
    idx_slices = np.arange(0, L, config.patch_len)

    output_predictions = {}
    inputs = []

    for t in idx_slices:
        inputs.append(S[:, t:t+config.patch_len][:, :, np.newaxis])

    input_data = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size=32)
    print("\n Prediction starts now...")
    p = model.predict(input_data, verbose=1)
    # import pdb;pdb.set_trace()

    output_prediction = np.hstack(p)

    return output_prediction

def get_single_prediction_hcqt(model, input_fname):

    S = compute_hcqt(input_fname)

    L = S.shape[1] - config.patch_len
    idx_slices = np.arange(0, L, config.patch_len)

    output_predictions = {}
    inputs = []

    for t in idx_slices:
        inputs.append(S[:, t:t+config.patch_len, :])

    input_data = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size=32)
    print("\n Prediction starts now...")
    p = model.predict(input_data, verbose=1)
    # import pdb;pdb.set_trace()

    output_predictions["sop"] = np.hstack(p[0])
    output_predictions["alt"] = np.hstack(p[1])
    output_predictions["ten"] = np.hstack(p[2])
    output_predictions["bas"] = np.hstack(p[3])

    return output_predictions


def get_single_prediction_1d(model, input_fname):
    # we first deal with the scenario of having an audio as input
    if input_fname.endswith('wav'):
        x, _ = librosa.load(
            os.path.join(config.audio_save_folder, input_fname), sr=config.fs
        )

        S = feature_extraction(input_fname, x, save=False)

    # here we deal with the scenario where we have pre-computed features
    elif input_fname.endswith('npy'):
        S = np.load(input_fname)

    else:
        raise ValueError("Please specify either an audio file or a npy file"
                         "with the corresponding spectrogram.")

    R, C = S.shape  # check that 1 is the right dimension, should be time
    ## zeropad for centered frames

    side = int(config.patch_len / 2)
    zp_spec = np.zeros([R, C + config.patch_len])

    ZR, ZC = zp_spec.shape

    zp_spec[:, side:(ZC - side)] = S

    idx_slices = np.arange(side, ZC - side)

    output_predictions = {}
    output_predictions['sop'] = []
    output_predictions['alt'] = []
    output_predictions['ten'] = []
    output_predictions['bas'] = []
    inputs = []

    for t in idx_slices:
        # input_spec = zp_spec[:, t-side:t-side+config.patch_len][np.newaxis, :, :, np.newaxis]
        inputs.append(zp_spec[:, t - side:t - side + config.patch_len][np.newaxis:, :, np.newaxis])
        # p = model(input_spec, training=False)
        # output_predictions['sop'].append(p[0][0])
        # output_predictions['alt'].append(p[1][0])
        # output_predictions['ten'].append(p[2][0])
        # output_predictions['bas'].append(p[3][0])

    input_data = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size=32)
    # inputs = np.array(inputs)
    print("\n Prediction starts now...")
    p = model.predict(input_data, verbose=1)
    # import pdb;pdb.set_trace()

    output_predictions["sop"] = p[0]
    output_predictions["alt"] = p[1]
    output_predictions["ten"] = p[2]
    output_predictions["bas"] = p[3]

    return output_predictions

def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert pitch activation map to multipitch
    by peak picking and thresholding
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    #
    # for f, t in zip(idx[0], idx[1]):
    #
    #     if np.array(f).ndim > 1:
    #         idx_max = peak_thresh_mat[t, f].argmax()
    #         est_freqs[t] = freqs[f[idx]]
    #
    #     else:
    #         est_freqs[t] = freqs[f]


    return times.reshape(len(times),), est_freqs#.reshape(len(est_freqs),)

def pitch_activations_to_f0(pitch_activation_mat, thresh):
    """Convert pitch activation vector to pitch traj
    """
    if pitch_activation_mat.shape[-1] == 1:
        pitch_activation_mat = pitch_activation_mat[:,:,0]

    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    est_freqs = np.zeros((pitch_activation_mat.shape[1], 1))

    for i, frame in enumerate(peak_thresh_mat.transpose()):

        max_idx = np.argmax(frame)

        # import pdb; pdb.set_trace()

        if peak_thresh_mat[max_idx, i] >= thresh:
            est_freqs[i] = freqs[max_idx]

    return times.reshape(len(times),), est_freqs.reshape(len(est_freqs),)

def pitch_activations_to_f0_argmax(pitch_activation_mat, thresh):
    """Convert pitch activation map to pitch by argmaxing
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = np.argmax(pitch_activation_mat, axis=0)
    for i in range(peak_thresh_mat.shape[1]):
        peak_thresh_mat[peaks[i], i] = pitch_activation_mat[peaks[i], i]

    idx = np.where(peak_thresh_mat >= thresh)

    #est_freqs = [[] for _ in range(len(times))]
    est_freqs = np.zeros(len(times))

    for f, t in zip(idx[0], idx[1]):
        if f == 0:
            est_freqs[t] = 0
        else:
            if np.array(f).size > 1:
                idx_max = peak_thresh_mat[t, f].argmax()
                est_freqs[t] = freqs[f[idx_max]]
            else:
                est_freqs[t] = freqs[f]

    # est_freqs = [np.array(lst) for lst in est_freqs]

    #return times.reshape(len(times),), est_freqs.reshape(len(est_freqs),)
    return times, est_freqs


def save_multif0_output(times, freqs, output_path):
    """save multif0 output to a csv file
    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)

def load_multipitch_annotations(mtracks, fname):

    annotation_files = mtracks[fname]['annot_files']
    annotation_dir = mtracks[fname]['annot_folder']

    references = []
    max_len = 0
    max_idx = 0
    for i in range(len(annotation_files)):

        d = jams.load(os.path.join(annotation_dir, annotation_files[i]))
        data = np.array(d.annotations[0].data)[:, [0, 2]]
        ref_times = data[:, 0].astype(float)
        ref_times = ref_times.reshape(-1, )

        # if i == 0:
        #     references.append(ref_times)

        ref_freqs = []
        for d in data[:, 1]:
            ref_freqs.append(d['frequency'])
        ref_freqs = np.array(ref_freqs).reshape(-1, )

        if len(ref_times) > max_len:
            max_len = len(ref_times)
            max_idx = i

        references.append(
            [ref_times, ref_freqs]
        )

    # resampling all curves to the longest time
    ref_times = references[max_idx][0]
    reference_multipitch = np.zeros([max_len, 5])
    reference_multipitch[:, 0] = ref_times

    for i in range(4):
        freqs, voicing = mir_eval.melody.freq_to_voicing(references[i][1])
        try:
            resampled_freqs, _ = mir_eval.melody.resample_melody_series(references[i][0], freqs, voicing, ref_times,
                                                                    kind='nearest')
        except:
            return None

        reference_multipitch[:, i + 1] = resampled_freqs

    # references = np.hstack(references)

    return reference_multipitch

def load_individual_annotations(mtracks, fname):

    annotation_files = mtracks[fname]['annot_files']
    annotation_dir = mtracks[fname]['annot_folder']

    references = []
    max_len = 0
    max_idx = 0
    for i in range(len(annotation_files)):

        d = jams.load(os.path.join(annotation_dir, annotation_files[i]))
        data = np.array(d.annotations[0].data)[:, [0, 2]]
        ref_times = data[:, 0].astype(float)
        ref_times = ref_times.reshape(-1, )

        # if i == 0:
        #     references.append(ref_times)

        ref_freqs = []
        for d in data[:, 1]:
            ref_freqs.append(d['frequency'])
        ref_freqs = np.array(ref_freqs).reshape(-1, )

        if len(ref_times) > max_len:
            max_len = len(ref_times)
            max_idx = i

        references.append(
            [ref_times, ref_freqs]
        )

    # # resampling all curves to the longest time
    # ref_times = references[max_idx][0]
    # reference_multipitch = np.zeros([max_len, 5])
    # reference_multipitch[:, 0] = ref_times
    #
    # for i in range(4):
    #     freqs, voicing = mir_eval.melody.freq_to_voicing(references[i][1])
    #     try:
    #         resampled_freqs, _ = mir_eval.melody.resample_melody_series(references[i][0], freqs, voicing, ref_times,
    #                                                                 kind='nearest')
    #     except:
    #         return None
    #
    #     reference_multipitch[:, i + 1] = resampled_freqs
    #
    # # references = np.hstack(references)

    return references

def load_multipitch_predictions(f0_predictions):

    max_len = 0

    # check if they have different sizes for resampling. Probably unnecessary because predictions should have
    # the correct shape

    if not len(f0_predictions[0]) == len(f0_predictions[2]):

        for i in range(len(f0_predictions)):
            if len(f0_predictions[i]) > max_len:
                max_len = len(f0_predictions[i])
                max_idx = i

        # resampling all curves to the longest time
        ref_times = get_time_grid(len(f0_predictions[max_idx]))
        #predictions.append(ref_times)

        reference_multipitch = np.zeros([max_len, 5])
        reference_multipitch[:, 0] = ref_times

        for i in range(len(f0_predictions)):

            freqs, voicing = mir_eval.melody.freq_to_voicing(f0_predictions[i])
            temp_times = get_time_grid(len(f0_predictions[i]))


            resampled_freqs, _ = mir_eval.melody.resample_melody_series(temp_times, freqs, voicing, ref_times,
                                                                        kind='nearest')

            reference_multipitch[:, i + 1] = resampled_freqs

    else:
        reference_multipitch = np.zeros([len(f0_predictions[0]), 5])
        ref_times = get_time_grid(len(f0_predictions[0]))
        reference_multipitch[:, 0] = ref_times
        for i in range(4):
            reference_multipitch[:, i+1] = f0_predictions[i]

    return reference_multipitch


    # #import pdb; pdb.set_trace()
    #
    # for i in range(len(f0_predictions)):
    #
    #     predictions.append(f0_predictions[i].reshape(len(f0_predictions[i]), 1))
    # try:
    #     predictions = np.hstack(predictions)
    #     return predictions
    #
    # except:
    #     print("Sizes of predictions and time vector do not match. Resampling to the longest sequence...")


def optimize_threshold(validation_files, mtracks, model, name):
    '''Optimize detection threshold on the validation set according to multpitch metrics.
    We select one single threshold for all voices
    '''

    for fname in validation_files:

        # load multipitch reference and arrange for eval
        reference = load_multipitch_annotations(mtracks, fname)

        if reference is None:
            continue

        ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])

        # get rid of zeros in reference for input to mir_eval
        for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            if any(fqs == 0):
                ref_freqs[i] = np.array([f for f in fqs if f > 0])


        # predict salience for each voice
        pth_npy = os.path.join(config.feat_targ_folder, 'inputs', fname.replace('wav', 'npy'))
        prediction_mat = get_single_prediction(model, pth_npy)

        thresh_vals = np.arange(0.1, 1.0, 0.1)
        thresh_scores = {t: [] for t in thresh_vals}

        for thresh in thresh_vals:

            _, sop = pitch_activations_to_mf0(prediction_mat['sop'], thresh=thresh)
            _, alt = pitch_activations_to_mf0(prediction_mat['alt'], thresh=thresh)
            _, ten = pitch_activations_to_mf0(prediction_mat['ten'], thresh=thresh)
            _, bas = pitch_activations_to_mf0(prediction_mat['bas'], thresh=thresh)

            predictions = load_multipitch_predictions([sop, alt, ten, bas])

            if predictions is not None:

                est_times, est_freqs = predictions[:, 0], list(predictions[:, 1:])

                # get rid of zeros in prediction for input to mir_eval
                for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                    if any(fqs == 0):
                        est_freqs[i] = np.array([f for f in fqs if f > 0])

                try:
                    metrics = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, max_freq=9000.0)

                except ValueError:
                    continue

                thresh_scores[thresh].append(metrics['Accuracy'])

            else:
                continue

    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]

    try:
        with open("{}_threshold.csv".format(name), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Best threshold for joint optimization of all voices is: {}".format(best_thresh)])

    except:
        print("Best threshold is {}".format(best_thresh))

    return best_thresh

def optimize_threshold_individual(validation_files, mtracks, model, name):
    '''Optimize detection threshold on the validation set according to multpitch metrics.
    We select one single threshold for all voices
    '''

    for fname in validation_files:

        # load multipitch reference and arrange for eval
        reference = load_individual_annotations(mtracks, fname)

        if reference is None:
            continue

        ref_times_sop, ref_freqs_sop = reference[0][0], reference[0][1]
        ref_times_alt, ref_freqs_alt = reference[1][0], reference[1][1]
        ref_times_ten, ref_freqs_ten = reference[2][0], reference[2][1]
        ref_times_bass, ref_freqs_bass = reference[3][0], reference[3][1]

        # predict salience for each voice
        if "hcqt" in name:
            pth_audio = os.path.join(mtracks[fname]["audiopath"], fname)
            prediction_mat = get_single_prediction_hcqt(model, pth_audio)

        else:
            pth_npy = os.path.join(config.feat_targ_folder, 'inputs', fname.replace('wav', 'npy'))
            prediction_mat = get_single_prediction(model, pth_npy)

        thresh_vals = np.arange(0.1, 1.0, 0.1)
        thresh_scores_sop = {t: [] for t in thresh_vals}
        thresh_scores_alt = {t: [] for t in thresh_vals}
        thresh_scores_ten = {t: [] for t in thresh_vals}
        thresh_scores_bass = {t: [] for t in thresh_vals}


        for thresh in thresh_vals:

            sop_time, sop = pitch_activations_to_f0(prediction_mat['sop'], thresh=thresh)
            alt_time, alt = pitch_activations_to_f0(prediction_mat['alt'], thresh=thresh)
            ten_time, ten = pitch_activations_to_f0(prediction_mat['ten'], thresh=thresh)
            bas_time, bas = pitch_activations_to_f0(prediction_mat['bas'], thresh=thresh)

            # predictions = load_multipitch_predictions([sop, alt, ten, bas])

            metrics_sop = mir_eval.melody.evaluate(ref_times_sop, ref_freqs_sop, sop_time, sop)
            metrics_alt = mir_eval.melody.evaluate(ref_times_alt, ref_freqs_alt, alt_time, alt)
            metrics_ten = mir_eval.melody.evaluate(ref_times_ten, ref_freqs_ten, ten_time, ten)
            metrics_bass = mir_eval.melody.evaluate(ref_times_bass, ref_freqs_bass, bas_time, bas)


            thresh_scores_sop[thresh].append(metrics_sop['Raw Pitch Accuracy'])
            thresh_scores_alt[thresh].append(metrics_alt['Raw Pitch Accuracy'])
            thresh_scores_ten[thresh].append(metrics_ten['Raw Pitch Accuracy'])
            thresh_scores_bass[thresh].append(metrics_bass['Raw Pitch Accuracy'])


    avg_thresh_sop = [np.mean(thresh_scores_sop[t]) for t in thresh_vals]
    avg_thresh_alt = [np.mean(thresh_scores_alt[t]) for t in thresh_vals]
    avg_thresh_ten = [np.mean(thresh_scores_ten[t]) for t in thresh_vals]
    avg_thresh_bass = [np.mean(thresh_scores_bass[t]) for t in thresh_vals]

    best_thresh_sop = thresh_vals[np.argmax(avg_thresh_sop)]
    best_thresh_alt = thresh_vals[np.argmax(avg_thresh_alt)]
    best_thresh_ten = thresh_vals[np.argmax(avg_thresh_ten)]
    best_thresh_bass = thresh_vals[np.argmax(avg_thresh_bass)]

    try:
        with open("{}_threshold.csv".format(name), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Best thresholds for separate optimizations of all voices are: S={}, A={}, T={}, B={}".format(
                best_thresh_sop,
                best_thresh_alt,
                best_thresh_ten,
                best_thresh_bass
            )])

    except:
        print("Best thresholds are {}".format([best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass]))

    return best_thresh_sop, best_thresh_alt, best_thresh_ten, best_thresh_bass

def optimize_threshold_latedeep(validation_files, mtracks, model, name):
    '''Optimize detection threshold on the validation set according to multpitch metrics.
    We select one single threshold for all voices. Output of Late/deep CNN with CQT
    '''

    for fname in validation_files:

        # load multipitch reference and arrange for eval
        reference = load_multipitch_annotations(mtracks, fname)

        if reference is None:
            continue

        ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])

        # get rid of zeros in reference for input to mir_eval
        for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            if any(fqs == 0):
                ref_freqs[i] = np.array([f for f in fqs if f > 0])


        # predict salience for each voice
        pth_npy = os.path.join(config.feat_targ_folder, 'inputs', fname.replace('wav', 'npy'))
        prediction_mat = get_single_prediction_ld(model, pth_npy)

        thresh_vals = np.arange(0.1, 1.0, 0.1)
        thresh_scores = {t: [] for t in thresh_vals}

        for thresh in thresh_vals:

            est_times, est_freqs = pitch_activations_to_mf0(prediction_mat, thresh=thresh)
            # est_times, est_freqs = list(est_times), list(est_freqs)
            # est_times = list(est_times)

            # get rid of zeros in prediction for input to mir_eval
            for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                if any(fqs == 0):
                    est_freqs[i] = np.array([f for f in fqs if f > 0])

            try:
                # import pdb; pdb.set_trace()
                metrics = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, max_freq=9000.0)

            except ValueError:
                continue

            thresh_scores[thresh].append(metrics['Accuracy'])


    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]

    try:
        with open("{}_threshold.csv".format(name), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Best threshold for joint optimization of all voices is: {}".format(best_thresh)])

    except:
        print("Best threshold is {}".format(best_thresh))

    return best_thresh

def eval_on_test_set(model, thresh, test_files, mtracks, name, save=False):

    # # read test set files
    # data_splits = load_json_data(os.path.join(
    #     config.feat_targ_folder, splits)
    # )
    # mtracks = load_json_data('mtracks_info.json')
    #
    # test_files = data_splits['test']


    all_metrics = []
    cnt = 0
    for fname in test_files:

        # progress viz
        cnt+=1
        progress(cnt, len(test_files))

        # load multipitch reference and arrange for eval
        reference = load_multipitch_annotations(mtracks, fname)

        if reference is None:
            continue

        ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])

        # get rid of zeros in reference for input to mir_eval
        for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            if any(fqs == 0):
                ref_freqs[i] = np.array([f for f in fqs if f > 0])


        # predict salience for each voice
        pth_npy = os.path.join(config.feat_targ_folder, 'inputs', fname.replace('wav', 'npy'))
        prediction_mat = get_single_prediction_ld(model, pth_npy)

        est_times, est_freqs = pitch_activations_to_mf0(prediction_mat, thresh=thresh)

        # get rid of zeros in prediction for input to mir_eval
        for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
            if any(fqs == 0):
                est_freqs[i] = np.array([f for f in fqs if f > 0])


        try:
            metrics = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
            metrics['track'] = fname

            all_metrics.append(metrics)

            if save:
                # save npy and freqs
                np.save(os.path.join(config.results_save_dir, "{}_prediction.npy".format(fname)), prediction_mat)

        except ValueError:
            continue

    pd.DataFrame(all_metrics).to_csv("{}_{}_all_scores.csv".format(name, 'testset'))

def eval_on_test_set_latedeep(model, thresh, test_files, mtracks, name, save=False):

    all_metrics = []
    cnt = 0
    for fname in test_files:

        if not fname.startswith("2_"): continue

        # progress viz
        cnt+=1
        progress(cnt, len(test_files))

        #####
        # load multipitch reference and arrange for eval
        reference = load_multipitch_annotations(mtracks, fname)

        if reference is None:
            continue

        ref_times, ref_freqs = reference[:, 0], list(reference[:, 1:])

        # get rid of zeros in reference for input to mir_eval
        for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            if any(fqs == 0):
                ref_freqs[i] = np.array([f for f in fqs if f > 0])


        # predict salience for each voice
        pth_npy = os.path.join(config.feat_targ_folder, 'inputs', fname.replace('wav', 'npy'))
        prediction_mat = get_single_prediction_ld(model, pth_npy)

        est_times, est_freqs = pitch_activations_to_mf0(prediction_mat, thresh=thresh)

        # get rid of zeros in prediction for input to mir_eval
        for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
            if any(fqs == 0):
                est_freqs[i] = np.array([f for f in fqs if f > 0])
        try:
            metrics = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, max_freq=9000.0)
            metrics['track'] = fname

            all_metrics.append(metrics)

            if save:
                # save npy and freqs
                np.save(os.path.join(config.results_save_dir, "{}_prediction_s.npy".format(fname)), prediction_mat['sop'])
                np.save(os.path.join(config.results_save_dir, "{}_prediction_a.npy".format(fname)), prediction_mat['alt'])
                np.save(os.path.join(config.results_save_dir, "{}_prediction_t.npy".format(fname)), prediction_mat['ten'])
                np.save(os.path.join(config.results_save_dir, "{}_prediction_b.npy".format(fname)), prediction_mat['bas'])

                times = get_time_grid(len(sop))
                pd.DataFrame(
                    np.hstack([times, sop])).to_csv(
                    os.path.join(
                        config.results_save_dir, "{}_f0_s.csv".format(fname)
                    ), header=False, index_label=False, index=False
                )
                pd.DataFrame(
                    np.hstack([times, alt])).to_csv(
                    os.path.join(
                        config.results_save_dir, "{}_f0_a.csv".format(fname)
                    ), header=False, index_label=False, index=False
                )
                pd.DataFrame(
                    np.hstack([times, ten])).to_csv(
                    os.path.join(
                        config.results_save_dir, "{}_f0_t.csv".format(fname)
                    ), header=False, index_label=False, index=False
                )
                pd.DataFrame(
                    np.hstack([times, bas])).to_csv(
                    os.path.join(
                        config.results_save_dir, "{}_f0_b.csv".format(fname)
                    ), header=False, index_label=False, index=False
                )
        except ValueError:
            continue

    pd.DataFrame(all_metrics).to_csv("{}_{}_all_scores.csv".format(name, 'testset'))

def eval_on_test_set_individual(model, thresholds, test_files, mtracks, name, save=True):


    all_metrics = []
    cnt = 0
    skipped = []
    for fname in test_files:

        cnt += 1
        progress(cnt, len(test_files))

        if not fname.startswith("2_"): continue

        # load multipitch reference and arrange for eval
        reference = load_individual_annotations(mtracks, fname)

        if reference is None:
            continue

        ref_times_sop, ref_freqs_sop = reference[0][0], reference[0][1]
        ref_times_alt, ref_freqs_alt = reference[1][0], reference[1][1]
        ref_times_ten, ref_freqs_ten = reference[2][0], reference[2][1]
        ref_times_bass, ref_freqs_bass = reference[3][0], reference[3][1]

        # predict salience for each voice
        if "hcqt" in name:
            pth_audio = os.path.join(mtracks[fname]["audiopath"], fname)
            prediction_mat = get_single_prediction_hcqt(model, pth_audio)

        else:
            pth_npy = os.path.join(config.feat_targ_folder, 'inputs', fname.replace('wav', 'npy'))
            prediction_mat = get_single_prediction(model, pth_npy)


        prediction_sop = prediction_mat["sop"]
        prediction_alt = prediction_mat["alt"]
        prediction_ten = prediction_mat["ten"]
        prediction_bass = prediction_mat["bas"]



        try:
            sop_time, sop = pitch_activations_to_f0(prediction_sop, thresh=thresholds[0])
            alt_time, alt = pitch_activations_to_f0(prediction_alt, thresh=thresholds[1])
            ten_time, ten = pitch_activations_to_f0(prediction_ten, thresh=thresholds[2])
            bas_time, bas = pitch_activations_to_f0(prediction_bass, thresh=thresholds[3])

            sop, alt, ten, bas = list(sop), list(alt), list(ten), list(bas)

            ### MPE evaluation

            mpe_est_times = sop_time
            mpe_est_freqs = list(np.vstack((sop, alt, ten, bas)).transpose())

            for i, (tms, fqs) in enumerate(zip(mpe_est_times, mpe_est_freqs)):
                if any(fqs == 0):
                    mpe_est_freqs[i] = np.array([f for f in fqs if f > 0])

            # import pdb; pdb.set_trace()
            if np.array(np.unique([len(ref_freqs_sop), len(ref_freqs_alt), len(ref_freqs_ten), len(ref_freqs_bass)])).size > 1:

                min_len = np.min([len(sop), len(alt), len(ten), len(bas)])
                ref_times_sop, ref_freqs_sop = ref_times_sop[:min_len], ref_freqs_sop[:min_len]
                ref_times_alt, ref_freqs_alt = ref_times_ten[:min_len], ref_freqs_alt[:min_len]
                ref_times_ten, ref_freqs_ten = ref_times_ten[:min_len], ref_freqs_ten[:min_len]
                ref_times_bass, ref_freqs_bass = ref_times_bass[:min_len], ref_freqs_bass[:min_len]
                mpe_ref_times = ref_times_sop[:min_len]
            else:
                mpe_ref_times = ref_times_sop

            mpe_ref_freqs = list(np.vstack((ref_freqs_sop, ref_freqs_alt, ref_freqs_ten, ref_freqs_bass)).transpose())

            for i, (tms, fqs) in enumerate(zip(mpe_ref_times, mpe_ref_freqs)):
                if any(fqs == 0):
                    mpe_ref_freqs[i] = np.array([f for f in fqs if f > 0])

            mpe_metrics = mir_eval.multipitch.evaluate(mpe_ref_times, mpe_ref_freqs, mpe_est_times, mpe_est_freqs)
            mpe_metrics["F-Score"] = 2 * (mpe_metrics["Precision"]*mpe_metrics["Recall"])/(mpe_metrics["Precision"]+mpe_metrics["Recall"])
            mpe_metrics["voice"] = "MPE"
            mpe_metrics["track"] = fname
            all_metrics.append(mpe_metrics)
            ref_freqs_sop = list(ref_freqs_sop)

            ### SOPRANO evaluation
            for i, (tms, fqs) in enumerate(zip(ref_times_sop, ref_freqs_sop)):
                if fqs == 0:
                    ref_freqs_sop[i] = np.array([])
                else:
                    ref_freqs_sop[i] = np.array([fqs])
            for i, (tms, fqs) in enumerate(zip(sop_time, sop)):
                if fqs == 0:
                    sop[i] = np.array([])
                else:
                    sop[i] = np.array([fqs])

            sop_metrics = mir_eval.multipitch.evaluate(ref_times_sop, ref_freqs_sop, sop_time, sop)
            sop_metrics["F-Score"] = 2 * (sop_metrics["Precision"]*sop_metrics["Recall"])/(sop_metrics["Precision"]+sop_metrics["Recall"])
            sop_metrics["voice"] = "SOPRANO"
            sop_metrics["track"] = fname
            all_metrics.append(sop_metrics)

            ### ALTO evaluation
            ref_freqs_alt = list(ref_freqs_alt)
            for i, (tms, fqs) in enumerate(zip(ref_times_alt, ref_freqs_alt)):
                if fqs == 0:
                    ref_freqs_alt[i] = np.array([])
                else:
                    ref_freqs_alt[i] = np.array([fqs])

            for i, (tms, fqs) in enumerate(zip(alt_time, alt)):
                if fqs == 0:
                    alt[i] = np.array([])
                else:
                    alt[i] = np.array([fqs])

            alt_metrics = mir_eval.multipitch.evaluate(ref_times_alt, ref_freqs_alt, alt_time, alt)
            alt_metrics["F-Score"] = 2 * (alt_metrics["Precision"]*alt_metrics["Recall"])/(alt_metrics["Precision"]+alt_metrics["Recall"])
            alt_metrics["voice"] = "ALTO"
            alt_metrics["track"] = fname
            all_metrics.append(alt_metrics)

            ### TENOR evaluation
            ref_freqs_ten = list(ref_freqs_ten)
            for i, (tms, fqs) in enumerate(zip(ref_times_ten, ref_freqs_ten)):
                if fqs == 0:
                    ref_freqs_ten[i] = np.array([])
                else:
                    ref_freqs_ten[i] = np.array([fqs])
            for i, (tms, fqs) in enumerate(zip(ten_time, ten)):
                if fqs == 0:
                    ten[i] = np.array([])
                else:
                    ten[i] = np.array([fqs])

            ten_metrics = mir_eval.multipitch.evaluate(ref_times_ten, ref_freqs_ten, ten_time, ten)
            ten_metrics["F-Score"] = 2 * (ten_metrics["Precision"]*ten_metrics["Recall"])/(ten_metrics["Precision"]+ten_metrics["Recall"])
            ten_metrics["voice"] = "TENOR"
            ten_metrics["track"] = fname
            all_metrics.append(ten_metrics)

            ### BASS evaluation
            ref_freqs_bass = list(ref_freqs_bass)
            for i, (tms, fqs) in enumerate(zip(ref_times_bass, ref_freqs_bass)):
                if fqs == 0:
                    ref_freqs_bass[i] = np.array([])
                else:
                    ref_freqs_bass[i] = np.array([fqs])
            for i, (tms, fqs) in enumerate(zip(bas_time, bas)):
                if fqs == 0:
                    bas[i] = np.array([])
                else:
                    bas[i] = np.array([fqs])

            bass_metrics = mir_eval.multipitch.evaluate(ref_times_bass, ref_freqs_bass, bas_time, bas)
            bass_metrics["F-Score"] = 2 * (bass_metrics["Precision"]*bass_metrics["Recall"])/(bass_metrics["Precision"]+bass_metrics["Recall"])
            bass_metrics["voice"] = "BASS"
            bass_metrics["track"] = fname
            all_metrics.append(bass_metrics)





            if save:
                np.save(os.path.join(config.results_save_dir, "{}_prediction_s.npy".format(fname)),
                        prediction_mat['sop'])
                np.save(os.path.join(config.results_save_dir, "{}_prediction_a.npy".format(fname)),
                        prediction_mat['alt'])
                np.save(os.path.join(config.results_save_dir, "{}_prediction_t.npy".format(fname)),
                        prediction_mat['ten'])
                np.save(os.path.join(config.results_save_dir, "{}_prediction_b.npy".format(fname)),
                        prediction_mat['bas'])

                # pd.DataFrame(
                #     np.hstack([sop_time, sop])).to_csv(
                #     os.path.join(
                #         config.results_save_dir, "{}_f0_s.csv".format(fname)
                #     ), header=False, index_label=False, index=False
                # )
                # pd.DataFrame(
                #     np.hstack([alt_time, alt])).to_csv(
                #     os.path.join(
                #         config.results_save_dir, "{}_f0_a.csv".format(fname)
                #     ), header=False, index_label=False, index=False
                # )
                # pd.DataFrame(
                #     np.hstack([ten_time, ten])).to_csv(
                #     os.path.join(
                #         config.results_save_dir, "{}_f0_t.csv".format(fname)
                #     ), header=False, index_label=False, index=False
                # )
                # pd.DataFrame(
                #     np.hstack([bas_time, bas])).to_csv(
                #     os.path.join(
                #         config.results_save_dir, "{}_f0_b.csv".format(fname)
                #     ), header=False, index_label=False, index=False
                # )
        except:
            skipped.append(fname)
            continue


    pd.DataFrame(all_metrics).to_csv("{}_{}_all_scores_individual_mpe.csv".format(name, 'testset'))
    print(skipped)
