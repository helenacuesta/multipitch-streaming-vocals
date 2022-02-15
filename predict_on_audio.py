"""
Predict four pitch trajectories given an input audio recording of a quartet mixture.
Models optimized for four-part a cappella ensemble singing.
"""

from __future__ import print_function
import models

import config

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import librosa
import scipy.signal

import os
import argparse


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

def feature_extraction(x):

    cqt = np.abs(librosa.cqt(
        y=x, sr=config.fs, hop_length=config.hopsize, fmin=config.f_min, n_bins=config.num_features, bins_per_octave=config.bins_per_octave
    ))

    cqt_db = librosa.amplitude_to_db(cqt)
    cqt_db = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min())

    return cqt_db

def get_single_prediction(model, input_fname):

    # we first deal with the scenario of having an audio as input
    if input_fname.endswith('wav'):
        x, _ = librosa.load(input_fname, sr=config.fs)
        S = feature_extraction(x)

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

    #est_freqs = [[] for _ in range(len(times))]
    est_freqs = np.zeros([len(times), 1])

    for f, t in zip(idx[0], idx[1]):

        if np.array(f).ndim > 1:
            idx_max = peak_thresh_mat[t, f].argmax()
            est_freqs[t] = freqs[f[idx_max]]

        else:
            est_freqs[t] = freqs[f]


    return times.reshape(len(times),1), est_freqs.reshape(len(est_freqs),1)


def main(args):

    model_name = args.model_name
    audiofile = args.audiofile
    audio_folder = args.audio_folder
    save = args.save

    # load model weights
    if model_name == 'unet_stand':

        save_key = 'unet_stand'
        model_path = "./models/unet_stand_cqt.h5"
        model = models.unet()
        model.load_weights(model_path)
        thresholds = [0.1, 0.2, 0.1, 0.1]

    elif model_name == 'unet_harm':

        save_key = 'unet_harm'
        model_path = "./models/unet_harm_cqt.h5"
        model = models.unet_harmonic()
        model.load_weights(model_path)
        thresholds = [0.1, 0.1, 0.1, 0.1]

    elif model_name == 'unet_harm_noskip':

        save_key = 'unet_harm_noskip'
        model_path = "./models/unet_harm_noskip.h5"
        model = models.unet_harmonic_noskip()
        model.load_weights(model_path)
        thresholds = [0.1, 0.1, 0.1, 0.1]

    elif model_name == 'unet_harm_noskip_nobsq':

        save_key = 'unet_harm_noskip_nobsq'
        model_path = "./models/unet_harm_noskip_nobsq.h5"
        model = models.unet_harmonic_noskip()
        model.load_weights(model_path)
        thresholds = [0.1, 0.1, 0.1, 0.1]

    elif model_name == 'unet_harm_nobsq':

        save_key = 'unet_harm_nobsq'
        model_path = "./models/unet_harm_cqt_nobsq.h5"
        model = models.unet_harmonic()
        model.load_weights(model_path)
        thresholds = [0.1, 0.1, 0.1, 0.1]

    elif model_name == 'unet_harm_hcqt':

        save_key = 'unet_harm_hcqt'
        model_path = "./models/unet_harm_hcqt.h5"
        model = models.unet_harmonic(hcqt_flag=True)
        model.load_weights(model_path)
        thresholds = [0.1, 0.1, 0.1, 0.1]

    elif model_name == 'unet_harm_hcqt_nobsq':

        save_key = 'unet_harm_hcqt_nobsq'
        model_path = "./models/unet_harm_hcqt_nobsq.h5"
        model = models.unet_harmonic(hcqt_flag=True)
        model.load_weights(model_path)
        thresholds = [0.1, 0.1, 0.1, 0.1]

    else:
        raise ValueError("Specified model must be unet_stand, unet_harm, unet_harm_noskip, unet_harm_noskip_nobsq, unet_harm_nobsq, unet_harm_hcqt, unet_harm_hcqt_nobsq.")


    output_dir = os.path.join(args.output_dir, save_key) #os.path.join("/home/helena/workspace/unet/results", save_key)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # compile model
    model.compile(
        optimizer=Adam(lr=config.init_lr),
        loss='mean_absolute_error',
        metrics=['mse']
    )
    print("Model compiled")

    # select operation mode and compute prediction
    if audiofile is not "0":

        fname = os.path.basename(audiofile)

        if "hcqt" in save_key:
            # predict using trained model
            predicted_output = get_single_prediction_hcqt(
                model, input_fname=audiofile
            )
        else:
            # predict using trained model
            predicted_output = get_single_prediction(
                model, input_fname=audiofile
            )

        prediction_sop = predicted_output["sop"]
        prediction_alt = predicted_output["alt"]
        prediction_ten = predicted_output["ten"]
        prediction_bass = predicted_output["bas"]

        sop_time, sop = pitch_activations_to_mf0(prediction_sop, thresh=thresholds[0])
        alt_time, alt = pitch_activations_to_mf0(prediction_alt, thresh=thresholds[1])
        ten_time, ten = pitch_activations_to_mf0(prediction_ten, thresh=thresholds[2])
        bas_time, bas = pitch_activations_to_mf0(prediction_bass, thresh=thresholds[3])

        pd.DataFrame(
            np.hstack([sop_time, sop])).to_csv(
            os.path.join(
                output_dir, "{}_f0_s.csv".format(fname.replace(".wav",""))
            ), header=False, index_label=False, index=False
        )

        pd.DataFrame(
            np.hstack([alt_time, alt])).to_csv(
            os.path.join(
                output_dir, "{}_f0_a.csv".format(fname.replace(".wav",""))
            ), header=False, index_label=False, index=False
        )
        pd.DataFrame(
            np.hstack([ten_time, ten])).to_csv(
            os.path.join(
                output_dir, "{}_f0_t.csv".format(fname.replace(".wav",""))
            ), header=False, index_label=False, index=False
        )
        pd.DataFrame(
            np.hstack([bas_time, bas])).to_csv(
            os.path.join(
                output_dir, "{}_f0_b.csv".format(fname.replace(".wav",""))
            ), header=False, index_label=False, index=False
        )

        if save:
            np.save(os.path.join(output_dir, "{}_prediction_s.npy".format(fname.replace(".wav",""))), prediction_sop)
            np.save(os.path.join(output_dir, "{}_prediction_a.npy".format(fname.replace(".wav",""))), prediction_alt)
            np.save(os.path.join(output_dir, "{}_prediction_t.npy".format(fname.replace(".wav",""))), prediction_ten)
            np.save(os.path.join(output_dir, "{}_prediction_b.npy".format(fname.replace(".wav",""))), prediction_bass)


        print(" > > > F0 predictions for {} have been successfully exported!".format(fname))

    elif audio_folder is not "0":

        for audiofile in os.listdir(audio_folder):

            if not audiofile.endswith('wav'): continue

            fname = audiofile

            if "hcqt" in save_key:
                # predict using trained model
                predicted_output = get_single_prediction_hcqt(
                    model, input_fname=os.path.join(audio_folder, audiofile)
                )
            else:
                # predict using trained model
                predicted_output = get_single_prediction(
                    model, input_fname=os.path.join(audio_folder, audiofile)
                )

            prediction_sop = predicted_output["sop"]
            prediction_alt = predicted_output["alt"]
            prediction_ten = predicted_output["ten"]
            prediction_bass = predicted_output["bas"]

            sop_time, sop = pitch_activations_to_mf0(prediction_sop, thresh=thresholds[0])
            alt_time, alt = pitch_activations_to_mf0(prediction_alt, thresh=thresholds[1])
            ten_time, ten = pitch_activations_to_mf0(prediction_ten, thresh=thresholds[2])
            bas_time, bas = pitch_activations_to_mf0(prediction_bass, thresh=thresholds[3])

            # import pdb;pdb.set_trace()

            pd.DataFrame(
                np.hstack([sop_time, sop])).to_csv(
                os.path.join(
                    output_dir, "{}_f0_s.csv".format(fname.replace(".wav",""))
                ), header=False, index_label=False, index=False
            )

            pd.DataFrame(
                np.hstack([alt_time, alt])).to_csv(
                os.path.join(
                    output_dir, "{}_f0_a.csv".format(fname.replace(".wav",""))
                ), header=False, index_label=False, index=False
            )
            pd.DataFrame(
                np.hstack([ten_time, ten])).to_csv(
                os.path.join(
                    output_dir, "{}_f0_t.csv".format(fname.replace(".wav",""))
                ), header=False, index_label=False, index=False
            )
            pd.DataFrame(
                np.hstack([bas_time, bas])).to_csv(
                os.path.join(
                    output_dir, "{}_f0_b.csv".format(fname.replace(".wav",""))
                ), header=False, index_label=False, index=False
            )

            if save:
                np.save(os.path.join(output_dir, "{}_prediction_s.npy".format(fname.replace(".wav", ""))),
                        prediction_sop)
                np.save(os.path.join(output_dir, "{}_prediction_a.npy".format(fname.replace(".wav",""))), prediction_alt)
                np.save(os.path.join(output_dir, "{}_prediction_t.npy".format(fname.replace(".wav",""))), prediction_ten)
                np.save(os.path.join(output_dir, "{}_prediction_b.npy".format(fname.replace(".wav",""))), prediction_bass)

            print(" > > > F0 predictions for {} have been successfully exported!".format(fname))
    else:
        raise ValueError("One of audiofile and audio_folder must be specified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output of an input audio file or all the audio files inside a folder.")

    parser.add_argument("--model",
                        dest='model_name',
                        type=str,
                        help="Specify the model to use for prediction:"
                             "unet_stand | unet_harm | unet_harm_noskip")

    parser.add_argument("--output",
                        dest='output_dir',
                        type=str,
                        help="Path to store the results.")

    parser.add_argument("--audiofile",
                        dest='audiofile',
                        default="0",
                        type=str,
                        help="Path to the audio file to analyze. If using the folder mode, this should be skipped.")

    parser.add_argument("--audiofolder",
                        dest='audio_folder',
                        default="0",
                        type=str,
                        help="Directory with audio files to analyze. If using the audiofile mode, this should be skipped.")
    parser.add_argument("--save",
                        dest="save",
                        default=True,
                        type=bool,
                        help="Boolean indicating whether to save output salience functions in addition to f0 contours.")

    main(parser.parse_args())


# python predict_on_audio.py --model unet_stand --audiofolder ../data/CantoriaDataset/mixes
# python predict_on_audio.py --model unet_harm --audiofolder ../data/CantoriaDataset/mixes
# python predict_on_audio.py --model unet_harm_noskip --audiofolder ../data/CantoriaDataset/mixes
#CUDA_VISIBLE_DEVICES=0 python predict_on_audio.py --model unet_harm_noskip_nobsq --audiofolder ../data/BQ/audiomix/
