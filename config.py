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
# old config file
'''
Config file
Dataset structure is created here
'''

import itertools
import numpy as np
import pandas as pd


'''Paths
'''
audio_save_folder = '/home/helena/workspace/data/audiomixtures/'
feat_targ_folder = '/home/helena/workspace/data/features_targets/'
exper_output = '/home/helena/workspace/data/experiment_output/'


# audio folders
csd_folder = '/home/helena/workspace/data/ChoralSingingDataset'

ecs_folder = '/home/helena/workspace/data/ECS'

dcs_folder_audio = '/home/helena/workspace/data/DCS/audio_wav_22050_mono/'
dcs_folder_annot = '/home/helena/workspace/data/DCS/annotations_csv_F0_PYIN/'

bc_folder_audio = '/home/helena/workspace/data/BC/stems'
bc_folder_annot = '/home/helena/workspace/data/BC/annotations/constant'

bsq_folder_audio = '/home/helena/workspace/data/BQ/stems'
bsq_folder_annot = '/home/helena/workspace/data/BQ/annotations/constant'

fs = 22050

'''All variables and parameters related to the dataset creation
'''

dataset = dict()
dataset['CSD'] = dict()
dataset['DCS'] = dict()
dataset['ECS'] = dict()
dataset['BC'] = dict()
dataset['BSQ'] = dict()

augmentation_idx = ['0_', '1_', '2_', '3_', '4_']

'''Choral Singing Dataset
'''
csd_songs = ['CSD_ER', 'CSD_LI', 'CSD_ND']

singers_csd = [
        'soprano_1', 'soprano_2', 'soprano_3', 'soprano_4',
        'alto_1', 'alto_2', 'alto_3', 'alto_4',
        'tenor_1', 'tenor_2', 'tenor_3', 'tenor_4',
        'bass_1', 'bass_2', 'bass_3', 'bass_4']

dataset['CSD']['songs'] = []
for song in csd_songs:
    for idx in augmentation_idx:
        dataset['CSD']['songs'].append(idx + song)

dataset['CSD']['singers'] = singers_csd

singers_per_section = 4
x = np.arange(1, singers_per_section + 1).astype(np.int32)
combos = [p for p in itertools.product(x, repeat=4)]
dataset['CSD']['combos'] = combos



'''ESMUC ChoralSet (divided by songs for convenience)
'''

''' Der Greis
'''

ecs_dg = ['DG_take1', 'DG_take2', 'DG_take3_mixed', 'DG_take4_mixed']

singers_ecs_dg = [
    'S1', 'S2', 'S3', 'S4',
    'A1', 'A2', 'A3',
    'T1', 'T2', 'T3',
    'B1', 'B2']

dataset['ECS']['DG_singers'] = singers_ecs_dg

dataset['ECS']['DG_songs'] = []
for song in ecs_dg:
    for idx in augmentation_idx:
        dataset['ECS']['DG_songs'].append(idx + song)

sop = np.arange(1, 4 + 1)
alto = np.arange(1, 3 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['DG_combos'] = combos

''' Die Himmel
'''

ecs_dh = ['DH1_take2', 'DH2_take1']

singers_ecs_dh = [
    'S1-1', 'S2-1', 'S3-2', 'S4-2', 'S5-2',
    'A1', 'A2',
    'T1-1', 'T2-1', 'T3-2',
    'B1', 'B2'
]

dataset['ECS']['DH_singers'] = singers_ecs_dh

dataset['ECS']['DH_songs'] = []
for song in ecs_dh:
    for idx in augmentation_idx:
        dataset['ECS']['DH_songs'].append(idx + song)

sop = np.arange(1, 5 + 1)
alto = np.arange(1, 2 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['DH_combos'] = combos


''' Seele Christi
'''

ecs_sc = ['SC1_take1', 'SC1_take2', 'SC1_take3_mixed', 'SC2_take1',
          'SC2_take2', 'SC2_take3_mixed', 'SC3_take1', 'SC3_take2_mixed']


singers_ecs_sc = [
    'S1', 'S2', 'S3', 'S4', 'S5',
    'A1', 'A2',
    'T1', 'T2', 'T3',
    'B1', 'B2'
]

dataset['ECS']['SC_singers'] = singers_ecs_sc

dataset['ECS']['SC_songs'] = []
for song in ecs_sc:
    for idx in augmentation_idx:
        dataset['ECS']['SC_songs'].append(idx + song)

sop = np.arange(1, 5 + 1)
alto = np.arange(1, 2 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['SC_combos'] = combos


'''Dagstuhl ChoirSet
'''

dcs_settings = ['All', 'QuartetA', 'QuartetB']

singers_QB = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']
singers_QA = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']
singers_all_dyn = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

'''dcs_songs = [
    'DLI_All_Take1_', 'DLI_All_Take2_', 'DLI_All_Take3_',
    'DLI_QuartetA_Take1_', 'DLI_QuartetA_Take2_', 'DLI_QuartetA_Take3_', 'DLI_QuartetA_Take4_', 'DLI_QuartetA_Take5_',
    'DLI_QuartetA_Take6_', 'DLI_QuartetB_Take1_', 'DLI_QuartetB_Take2_', 'DLI_QuartetB_Take2_', 'DLI_QuartetB_Take3_',
    'DLI_QuartetB_Take4_', 'DLI_QuartetB_Take5_']
'''

# no combos because these are quartets (inside the full choir)
dcs_songs_fc = ['DCS_LI_FullChoir_Take01', 'DCS_LI_FullChoir_Take02', 'DCS_LI_FullChoir_Take03']
dcs_singers_fc = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

dataset['DCS']['FC_singers'] = dcs_singers_fc
dataset['DCS']['FC_songs'] = []
for song in dcs_songs_fc:
    for idx in augmentation_idx:
        dataset['DCS']['FC_songs'].append(idx + song)



dcs_songs_qa = ['DCS_LI_QuartetA_Take01', 'DCS_LI_QuartetA_Take02', 'DCS_LI_QuartetA_Take03',
                              'DCS_LI_QuartetA_Take04', 'DCS_LI_QuartetA_Take05', 'DCS_LI_QuartetA_Take06']
dcs_singers_qa = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']

dataset['DCS']['QA_singers'] = dcs_singers_qa
dataset['DCS']['QA_songs'] = []

for song in dcs_songs_qa:
    for idx in augmentation_idx:
        dataset['DCS']['QA_songs'].append(idx + song)


dcs_songs_qb = ['DCS_LI_QuartetB_Take01', 'DCS_LI_QuartetB_Take02', 'DCS_LI_QuartetB_Take03',
               'DCS_LI_QuartetB_Take04', 'DCS_LI_QuartetB_Take05']
dcs_singers_qb = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

dataset['DCS']['QB_singers'] = dcs_singers_qb
dataset['DCS']['QB_songs'] = []
for song in dcs_songs_qb:
    for idx in augmentation_idx:
        dataset['DCS']['QB_songs'].append(idx + song)

''' Bach Chorales
'''
bc = pd.read_csv('/home/helena/workspace/data/BC_info.csv').values

dataset['BC']['songs'] = []
dataset['BC']['num_parts'] = []
idx=0
for song in bc[:, 0]:
    idx += 1
    basename = "{}_{}".format(idx, song)
    for aug_idx in augmentation_idx:
        dataset['BC']['songs'].append(aug_idx + basename)
        dataset['BC']['num_parts'].append(bc[idx-1, 6])

dataset['BC']['singers'] = [bc[0, 1], bc[0, 2], bc[0, 3], bc[0, 4]]
#dataset['BC']['num_parts'] = bc[:, 6]


'''Barbershop Quartets
'''
bq = pd.read_csv('/home/helena/workspace/data/BQ_info.csv').values

dataset['BSQ']['songs'] = []
dataset['BSQ']['num_parts'] = []
idx=0
for song in bq[:, 0]:
    idx += 1
    basename = "{}_{}".format(idx, song)
    for aug_idx in augmentation_idx:
        dataset['BSQ']['songs'].append(aug_idx + basename)
        dataset['BSQ']['num_parts'].append(bq[idx-1, 6])

dataset['BSQ']['singers'] = [bq[0, 1], bq[0, 2], bq[0, 3], bq[0, 4]]
#dataset['BSQ']['num_parts'] = bq[:, 6]
