# Multi-pitch Streaming of Vocal Quartets

<span style="color:green">**This repository is currently under preparation. Essential updates are reported below.**</span>.

This is the accompanying repository of Chapter 5 of the PhD dissertation:

Helena Cuesta. _Data-driven Pitch Content Description of Choral Singing Recordings_. Submitted, 2022.
Universitat Pompeu Fabra, Barcelona.


## Description

The proposed multi-pitch streaming models convert an input audio recording of a four-part vocal ensemble into four independent pitch contours, one for each melodic voice.

As described in the documentation, there are three main models available, although we strongly recomment the use of **U-Net-Harm** (`unet_harm`) as it obtains the best performances in our experiments.

## Prediction

Here's how to call the prediction script from the command line:

``python predict_on_audio.py --model unet_harm --output output_dir --audiofile input_mixture.wav``

* The `model` argument can be `unet_harm`, `unet_stand`, `unet_harm_noskip`, `unet_harm_hcqt`, each referring to a different model variant as described in the reference.
* The `output` argument denotes the output directory to store the results (salience functions and F0 contours). One CSV file for each output F0 trajectory and one NPY file with each output salience will be stored
in this directory inside a folder with the model's name.
* The `audiofile` argument should be the full path to the input audio file.

The code also allows predicting the output of multiple files, which should all be in the same folder indicated using the parameter `audiofolder` instead of `audiofile`.

## Current status of the repo
* **Feb 15th 2022**: the models are currently not available. While we update this issue, we kindly ask you to download the desired model(s) from
<a href="https://drive.google.com/drive/folders/1-P3MWlMeZstfeUqXloCxYP-e79URG89X?usp=sharing">this link</a> and place then in a folder named `models/` in the root repo folder before running the `predict_on_audio.py` script.
