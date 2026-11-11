import numpy as np
import pandas as pd
from pathlib import Path
from mtrf import TRF
import mne
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Define parameters
subjects = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19',
    'sub20', 'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27',
    'sub28', 'sub29', 'sub30', 'sub31', 'sub32', 'sub33', 'sub34', 'sub35',
    'sub36', 'sub37', 'sub38', 'sub39', 'sub40']

conditions = ['a1', 'a2', 'e1', 'e2']

i = 3

condition = conditions[i]

# Select feature set: 'standard' or 'target_num'
feature_type = 'standard'
predictors = ['combined', 'envelopes', 'phonemes']
model_type = predictors[0]

project_path = Path.cwd()
trf_path = project_path / 'data' / 'TRF' / feature_type / model_type
trf_path.mkdir(parents=True, exist_ok=True)


golumbic = False

if golumbic:
    z_axis= -1
    norm = 'joint'
else:
    z_axis = 1
    norm = 'sep'

lambdas_path = project_path / 'data' / 'TRF' / 'optimisation'


predictions_dict = {}
subjects_dict = {}

# Load channel names once
eeg_path = project_path / 'data' / 'sub10' / 'preprocessed'
eeg_filename = 'sub10_a1_preprocessed-raw.fif'
eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=False)
all_chs = eeg_file.info['ch_names']

tmin,tmax = -0.1, 1.0


def choose_predictor_idx(predictor=''):
    if model_type == 'combined':
        if predictor == 'envelopes':
            pred_idx = 0
        elif predictor == 'phonemes':
            pred_idx = 1
        else:
            raise ValueError(f'Unknown predictor: {predictor}')

    elif model_type == 'envelopes':
        pred_idx = 0

    elif model_type == 'phonemes':
        pred_idx = 0

    return pred_idx

for subject in subjects:

    features_path = project_path / 'data' / subject / 'features'

    # Select feature file
    if feature_type == 'standard':
        feature_filename = f'{subject}_{condition}_features.npz'
    elif feature_type == 'target_num':
        feature_filename = f'{subject}_{condition}_target_num_features.npz'
    else:
        raise ValueError(f'Unknown feature type: {feature_type}')

    features = np.load(features_path / feature_filename, allow_pickle=True)

    # Load envelopes, phonemes and responses
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

    # Build predictor matrices
    target_env, distr_env, target_phon, distr_phon, plane, target_stream = streams()
    columns = ['envelopes', 'phonemes', 'responses']


    def phoneme_scaling_factor(phon_arr):
        p = np.count_nonzero(phon_arr) / phon_arr.size
        return np.sqrt(p * (1 - p))

    target_phon_std = phoneme_scaling_factor(target_phon)
    distr_phon_std = phoneme_scaling_factor(distr_phon)

    if model_type == 'combined':
        targ_matrix = np.stack((target_env, target_phon, responses))
        distr_matrix = np.stack((distr_env, distr_phon, responses))
    elif model_type == 'envelopes':
        targ_matrix = np.stack((target_env, responses))
        distr_matrix = np.stack((distr_env, responses))
    elif model_type == 'phonemes':
        targ_matrix = np.stack((target_phon, responses))
        distr_matrix = np.stack((distr_phon, responses))

    # Load EEG
    eeg_path = project_path / 'data' / subject / 'preprocessed'
    eeg_filename = f'{subject}_{condition}_preprocessed-raw.fif'
    eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=True)

    if 'AFz' in eeg_file.ch_names:
        eeg_file.drop_channels('AFz')

    # Standardize across time within each EEG channel
    # EEG shape: channels x time; axis=1 is temporal z-scoring, -1 is joint multichannel z-scoring
    eeg_data = zscore(eeg_file.get_data(), axis=z_axis)

    subjects_dict[subject] = {'target': targ_matrix.T, 'distractor': distr_matrix.T, 'eeg': eeg_data.T}

    def run_trf(phon_std=None, predictor='', stream=''):
        pred_idx = choose_predictor_idx(predictor=predictor)
        matrix = subjects_dict[subject][stream]

        if golumbic == True:
            optimal_lambda = np.load(lambdas_path / 'subjects_lambdas' / f'{subject}.npz', allow_pickle=True)['lambdas'].item()
        else:
            optimal_lambda = np.load(lambdas_path / f'best_lambdas_{feature_type}.npz', allow_pickle=True)['lambdas'].item()
            # Collinearity check
        if model_type == 'combined':
            X = pd.DataFrame({'Envelope': matrix[:, 0], 'Phoneme': matrix[:, 1], 'Motor': matrix[:, 2]})
        elif model_type == 'envelopes':
            X = pd.DataFrame({'Envelope' : matrix[:, 0], 'Motor':matrix[:, 1]})
        elif model_type == 'phonemes':
            X = pd.DataFrame({'Phonemes': matrix[:, 0], 'Motor':matrix[:, 1]})

        corr = X.corr()
        vif = pd.DataFrame()
        vif['Predictor'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        collinearity_path = trf_path / 'collinearity'
        collinearity_path.mkdir(parents=True, exist_ok=True)

        output_file = collinearity_path / f'{condition}_{subject}_collinearity{norm}.xlsx'

        with pd.ExcelWriter(output_file) as writer:
            corr.to_excel(writer, sheet_name='Correlations')
            vif.to_excel(writer, sheet_name='VIF', index=False)

        eeg_data = subjects_dict[subject]['eeg']

        # Train forward TRF
        trf = TRF(direction=1, method='ridge')
        trf.train(stimulus=matrix, response=eeg_data, fs=125, tmin=tmin, tmax=tmax, regularization=optimal_lambda, average=True, seed=42)

        predictions, r = trf.predict(stimulus=matrix, response=eeg_data, average=False)

        weights = trf.weights[pred_idx, :, :]
        if predictor == 'phonemes':
            weights_std = weights * phon_std
        else:
            weights_std = weights

        # Smooth weights with Hamming window
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()

        smoothed_weights = np.empty_like(weights_std)

        for ch in range(weights.shape[-1]):
            smoothed_weights[:, ch] = np.convolve(weights_std[:, ch], hamming_win, mode='same')

        return smoothed_weights, r

    # Envelopes
    target_env_weights, target_r = run_trf(None, predictor='envelopes', stream='target')
    distr_env_weights, distr_r = run_trf(None, predictor='envelopes', stream='distractor')

    # Phonemes
    target_phon_weights, _ = run_trf(target_phon_std, predictor='phonemes', stream='target')
    distr_phon_weights, _ = run_trf(distr_phon_std, predictor='phonemes', stream='distractor')

    predictions_dict[subject] = {'target_env': target_env_weights, 'target_phon': target_phon_weights,
                                 'distr_env': distr_env_weights, 'distr_phon': distr_phon_weights,
                                 'target_accuracy': target_r, 'distractor_accuracy': distr_r}


# Stack TRF weights and prediction accuracies
def get_pred_data(predictions, predictor=''):
    target_data = []
    distr_data = []
    target_r_data = []
    distr_r_data = []

    for subs, weights in predictions.items():
        target_data.append(weights[f'target_{predictor}'])
        distr_data.append(weights[f'distr_{predictor}'])
        target_r_data.append(weights['target_accuracy'])
        distr_r_data.append(weights['distractor_accuracy'])

    return target_data, distr_data, target_r_data, distr_r_data


target_env_data, distr_env_data, target_r_data, distr_r_data = get_pred_data(predictions_dict, predictor='env')
target_phon_data, distr_phon_data, _, _ = get_pred_data(predictions_dict, predictor='phon')


# Save TRF weights and prediction accuracy in feature-specific folder
np.savez(trf_path / f'{condition}_pred_data_all_envelopes_{norm}.npz', target=target_env_data, distractor=distr_env_data)
np.savez(trf_path / f'{condition}_pred_data_all_phonemes_{norm}.npz', target=target_phon_data, distractor=distr_phon_data)
np.savez(trf_path / f'{condition}_pred_data_accuracy_vals_{norm}.npz', target=target_r_data, distractor=distr_r_data)


# Save matrices
def get_matrix_data():
    eeg_list = []
    target_list = []
    distr_list = []

    for subject in subjects_dict:
        eeg_data = subjects_dict[subject]['eeg']
        target_matrix = subjects_dict[subject]['target']
        distr_matrix = subjects_dict[subject]['distractor']

        eeg_list.append(eeg_data)
        target_list.append(target_matrix)
        distr_list.append(distr_matrix)

    return eeg_list, target_list, distr_list


eeg_list, target_list, distr_list = get_matrix_data()
np.savez(trf_path / f'{condition}_matrices_{norm}.npz', target=np.array(target_list, dtype='object'), distr=np.array(distr_list, dtype='object'), eeg=np.array(eeg_list, dtype='object'))

print(f'\nFinished: {condition} | Feature type: {feature_type} | Model type: {model_type} | Normalisation: {norm}\n')
print(f'Results saved to: {trf_path}')
