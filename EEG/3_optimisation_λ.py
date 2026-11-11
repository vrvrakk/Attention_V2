import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mtrf import TRF
from mtrf.stats import crossval, neg_mse
import mne
from scipy.stats import zscore

subjects = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19',
    'sub20', 'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27',
    'sub28', 'sub29', 'sub30', 'sub31', 'sub32', 'sub33','sub34', 'sub35',
    'sub36', 'sub37', 'sub38', 'sub39', 'sub40'] # sub33 and sub35 - a1_0 and a2_3 header and marker files missing...


conditions = ['a1', 'a2', 'e1', 'e2']
feature_type = 'standard' # standard or target_num
predictions_dict = {}

subjects_dict = {}

for subject in subjects:

    project_path = Path.cwd()
    features_path = project_path / 'data' / subject / 'features'
    trf_path = project_path / 'data' / 'TRF'
    eeg_path = project_path / 'data' / 'sub10' / 'preprocessed'
    eeg_filename = f'sub10_a1_preprocessed-raw.fif'
    eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=True)
    all_chs = eeg_file.info['ch_names']

    subjects_dict[subject] = {}

    for condition in conditions:
        # load envelopes and responses
        # Select feature file
        if feature_type == 'standard':
            feature_filename = f'{subject}_{condition}_features.npz'
        elif feature_type == 'target_num':
            feature_filename = f'{subject}_{condition}_target_num_features.npz'
        else:
            raise ValueError(f'Unknown feature type: {feature_type}')
        features = np.load(features_path/feature_filename, allow_pickle=True)
        env_arr1 = features['env1']
        env_arr2 = features['env2']
        phon1_arr = features['phon1']
        phon2_arr = features['phon2']
        responses = features['responses']

        def streams():
            if condition in ['a1', 'a2']:
                plane = 'azimuth'
            else:
                plane = 'elevation'
            if condition in ['a1', 'e1']:
                target_stream = 's1'
                target_env = zscore(env_arr1)
                distr_env = zscore(env_arr2)
                target_phon = phon1_arr
                distr_phon = phon2_arr
            else:
                target_stream = 's2'
                target_env = zscore(env_arr2)
                distr_env = zscore(env_arr1)
                target_phon = phon2_arr
                distr_phon = phon1_arr

            print(f'Target stream in condition {condition} is: {target_stream}')
            return target_env, distr_env, target_phon, distr_phon, plane, target_stream

        # build matrix
        target_env, distr_env, target_phon, distr_phon, plane, target_stream = streams()
        columns = ['envelopes', 'phonemes', 'responses']

        targ_matrix = np.stack((target_env, target_phon, responses))
        distr_matrix = np.stack((distr_env, distr_phon)) # responses only once

        # load eeg:
        eeg_path = project_path / 'data' / subject / 'preprocessed'
        eeg_filename = f'{subject}_{condition}_preprocessed-raw.fif'
        eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=True)
        if len(eeg_file.info['ch_names']) == 64:
            eeg_file.drop_channels('AFz')

        eeg_data = zscore(eeg_file.get_data(), axis=1) # axis = None --> joint multi-channel normalisation
        # axis = 1 -> across time within each channel

        subjects_dict[subject][condition] = {'target': targ_matrix.T, 'distractor': distr_matrix.T, 'eeg': eeg_data.T}


# Candidate lambdas
lambdas = np.logspace(-2, 6, 20)
fs = 125

X = []
Y = []

for subjects in subjects_dict.keys():
    target_concat = []
    distr_concat = []
    eeg_concat = []
    for condition in subjects_dict[subjects].keys():
        target = subjects_dict[subjects][condition]['target']
        distractor = subjects_dict[subjects][condition]['distractor']
        eeg = subjects_dict[subjects][condition]['eeg']
        target_concat.append(target)
        distr_concat.append(distractor)
        eeg_concat.append(eeg)
    target_concatenated = np.concatenate(target_concat, axis=0)
    distr_concatenated = np.concatenate(distr_concat, axis=0)
    eeg_concatenated = np.concatenate(eeg_concat, axis=0)

    combined = np.column_stack((target_concatenated[:, 0], target_concatenated[:, 1], target_concatenated[:, 2],
        distr_concatenated[:, 0], distr_concatenated[:, 1])) # motor arr included only once

    X.append(combined)
    Y.append(eeg_concatenated)

# Test lambdas
mean_mse_dict = {}

for lmbda in lambdas:
    fwd_trf = TRF(direction=1, metric=neg_mse)

    neg_mse_cv = crossval(fwd_trf, X, Y, fs=fs, tmin=0.0, tmax=0.5, regularization=lmbda, average=False, seed=42)

    # neg_mse -> ordinary positive MSE
    mse_cv = np.mean(np.asarray(neg_mse_cv) * -1) # from mtrfpy toolbox -> Bialas

    mean_mse_dict[lmbda] = mse_cv

optimal_lambda = min(mean_mse_dict, key=mean_mse_dict.get)

print(f'Optimal lambda for all subjects, across conditions for {feature_type}:', optimal_lambda)
print('Minimum mean MSE:', mean_mse_dict[optimal_lambda])

lambdas_path = trf_path / 'optimisation'
os.makedirs(lambdas_path, exist_ok=True)
np.savez(lambdas_path/f'best_lambdas_{feature_type}', lambdas=optimal_lambda)

plt.plot(list(mean_mse_dict.keys()), list(mean_mse_dict.values()))
plt.xlabel(f'Lambda | {feature_type.capitalize()}', size=16)
plt.ylabel('Mean cross-validated MSE', size=16)
plt.yticks(size=16)
plt.xticks(size=16)
plt.show()
plt.savefig(lambdas_path/f'mse_curve_{feature_type}.png', dpi=300)

