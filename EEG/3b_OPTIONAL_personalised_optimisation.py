
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne

from mtrf import TRF
from mtrf.stats import neg_mse
from scipy.stats import zscore


# BASED ON THE GOLUMBIC JAEGER APPROACH 2025


# PARAMETERS
subjects = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20', 'sub21', 'sub22', 'sub23',
    'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29', 'sub30', 'sub31', 'sub32', 'sub33', 'sub34', 'sub35',
    'sub36', 'sub37', 'sub38', 'sub39', 'sub40']

conditions = ['a1', 'a2', 'e1', 'e2']
feature_type = 'target_num'

fs = 125
block_duration = 120
block_samples = fs * block_duration

lambdas = np.logspace(-2, 6, 20)

project_path = Path.cwd()

output_path = (project_path / 'data' / 'TRF' / 'optimisation' / 'subjects_lambdas')
output_path.mkdir(parents=True, exist_ok=True)


# LOAD SUBJECT DATA AND SPLIT INTO BLOCKS
def load_subject_blocks(subject):

    X_blocks = []
    Y_blocks = []

    features_path = project_path / 'data' / subject / 'features'
    eeg_path = project_path / 'data' / subject / 'preprocessed'

    for condition in conditions:

        if feature_type == 'standard':
            feature_filename = f'{subject}_{condition}_features.npz'
        elif feature_type == 'target_num':
            feature_filename = f'{subject}_{condition}_target_num_features.npz'
        else:
            raise ValueError(f'Unknown feature type: {feature_type}')

        with np.load(features_path / feature_filename, allow_pickle=True) as features:
            env1 = np.asarray(features['env1'])
            env2 = np.asarray(features['env2'])
            phon1 = np.asarray(features['phon1'])
            phon2 = np.asarray(features['phon2'])
            responses = np.asarray(features['responses'])

        if condition in ['a1', 'e1']:
            target_env = zscore(env1)
            distractor_env = zscore(env2)
            target_phon = phon1
            distractor_phon = phon2
        else:
            target_env = zscore(env2)
            distractor_env = zscore(env1)
            target_phon = phon2
            distractor_phon = phon1

        # Joint model: 5 predictors, motor included once
        X = np.column_stack([target_env, target_phon, responses, distractor_env, distractor_phon])

        eeg_filename = f'{subject}_{condition}_preprocessed-raw.fif'
        raw = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=True, verbose=False)

        if 'AFz' in raw.ch_names:
            raw.drop_channels('AFz')

        Y = zscore(raw.get_data(), axis=None).T # joint multichannel norm

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f'{subject} {condition}: 'f'feature samples={X.shape[0]}, EEG samples={Y.shape[0]}')

        # Each 2-minute segment is a separate CV fold.
        # Keep the shorter final segment if present.
        for start in range(0, len(X), block_samples):

            end = min(start + block_samples, len(X))

            X_block = X[start:end]
            Y_block = Y[start:end]

            # Must be longer than the maximum TRF lag
            if len(X_block) <= int(0.5 * fs) + 1:
                print(f'Skipping extremely short block: ' f'{subject}, {condition}, {len(X_block)} samples')
                continue

            X_blocks.append(X_block)
            Y_blocks.append(Y_block)

    return X_blocks, Y_blocks



# LEAVE-ONE-BLOCK-OUT CROSS-VALIDATION
def evaluate_lambda(X_blocks, Y_blocks, lmbda):

    fold_mses = []

    for test_idx in range(len(X_blocks)):

        X_train = [X_blocks[i] for i in range(len(X_blocks)) if i != test_idx]

        Y_train = [Y_blocks[i] for i in range(len(Y_blocks)) if i != test_idx]

        X_test = X_blocks[test_idx]
        Y_test = Y_blocks[test_idx]

        trf = TRF(direction=1, method='ridge', metric=neg_mse)

        trf.train(stimulus=X_train, response=Y_train, fs=fs, tmin=0.0, tmax=0.5, regularization=lmbda, average=True, seed=42)

        # Predict on held-out block
        predictions, score = trf.predict(stimulus=X_test, response=Y_test, average=False)

        # Explicit MSE across time and channels
        mse = np.mean((Y_test - predictions) ** 2)
        fold_mses.append(mse)

    return np.mean(fold_mses)


# ============================================================
# OPTIMIZE EACH SUBJECT
# ============================================================

for subject in subjects:

    print(f'\n{"=" * 60}')
    print(f'Optimizing {subject}')
    print(f'{"=" * 60}')

    X_blocks, Y_blocks = load_subject_blocks(subject)

    print(f'Number of CV folds: {len(X_blocks)}')
    print(f'Block lengths: {[len(x) for x in X_blocks]}')

    if len(X_blocks) < 2:
        raise ValueError(f'{subject}: fewer than 2 valid blocks')

    mse_values = []

    for lmbda in lambdas:

        mse = evaluate_lambda(X_blocks, Y_blocks, lmbda)
        mse_values.append(mse)

        print(f'Lambda={lmbda:.6g} | MSE={mse:.6f}')

    mse_values = np.asarray(mse_values)

    best_idx = np.argmin(mse_values)
    optimal_lambda = float(lambdas[best_idx])

    print(f'\n{subject}: optimal lambda = {optimal_lambda:.6g}')
    print(f'Minimum MSE = {mse_values[best_idx]:.6f}')

    # Overwrites existing file.
    # Only the optimal lambda is saved.
    np.savez(output_path / f'{subject}.npz', lambdas=optimal_lambda)

print('\nFinished subject-specific lambda optimization.')
print(f'Files saved to: {output_path}')
