import configparser
import os
import pandas as pd
#

RANDOM_STATE = 42

if 'CONFIG_PATH' in os.environ.keys():
    config_path = os.environ['CONFIG_PATH']
else:
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.ini')

config = configparser.ConfigParser()

config.read(config_path)

results_save_dir = '/home/helena/workspace/data/results/unet_harm_noskip'


sig_process = config["signal_processing"]
fs = int(sig_process["fs"])
hopsize = int(sig_process["hopsize"])
framesize = int(sig_process["framesize"])
hoptime = float(hopsize/fs)
stft_features = int(framesize/2+1)
stft_size = int(sig_process["stft_size"])
bins_per_octave = int(sig_process["bins_per_octave"])
n_octaves = int(sig_process["n_octaves"])
over_sample = int(sig_process["over_sample"])
harmonics = [int(x) for x in sig_process["harmonics"].split(', ')]
f_min =float(sig_process["fmin"])
patch_len = int(sig_process["patch_len"])

params = config["params"]
batch_size = int(params["batch_size"])
train_split = float(params["train_split"])
#lstm_size = int(params["lstm_size"])
num_features = int(params["num_features"])
init_lr = float(params["init_lr"])
max_models_to_keep = int(params["max_models_to_keep"])
print_every = int(params["print_every"])
save_every = int(params["save_every"])
samples_per_file = int(params["samples_per_file"])
#assert batch_size%samples_per_file==0, "batch_size must be multiple of samples_per_file"
blur = bool(params["blur"])
blur_max_std = float(params["blur_max_std"])
num_epochs = int(params["num_epochs"])
batches_per_epoch = int(params["batches_per_epoch"])
validation_steps = int(params["validation_steps"])

model_save_path = '/home/helena/workspace/unet/models'

'''U-net parameters
'''
unet_num_layers = 6
unet_num_filters = [16, 32, 64, 128, 256, 512]
unet_kernel_size = [5, 5]
unet_vertical_kernel_size = [70, 5]
strides = [1, 2]
# input_shape = (int(stft_size/2), None, 1)
input_shape = (num_features, None, 1)
input_shape_3d = (num_features, None, 5)

def change_variable(path, variable, new_value):
    global config
    global config_path
    config[path][variable] = new_value
    with open(config_path, 'w') as configfile:
        config.write(configfile)

#
