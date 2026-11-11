
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pandas.conftest import tick_classes
from statsmodels.stats.multitest import multipletests

font_manager.fontManager.addfont(r"C:\Windows\Fonts\times.ttf")
plt.rcParams['font.family'] = 'Times New Roman'
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy.stats import t

# Parameters
conditions = ['a1', 'a2', 'e1', 'e2']
plane = 'azimuth'
predictor = 'phonemes'
model_type = 'combined' # envelopes, phonemes as other options

golumbic = False

if golumbic:
    norm = 'joint'
else:
    norm = 'sep'

# Select feature type: 'standard' or 'target_num'
feature_type = 'standard'

if plane == 'azimuth':
    condition = conditions[0:2]
elif plane == 'elevation':
    condition = conditions[2:]

project_path = Path.cwd()

# Separate input/output folders for each feature type
trf_path = project_path / 'data' / 'TRF' / feature_type / model_type
fig_path = project_path / 'data' / 'figures' / feature_type / model_type
fig_path.mkdir(parents=True, exist_ok=True)

# Load channel names
eeg_path = project_path / 'data' / 'sub10' / 'preprocessed'
eeg_filename = 'sub10_a1_preprocessed-raw.fif'
eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=False)
eeg_file.rename_channels({'ref': 'FCz'})
info = eeg_file.info

# Time vector
times = np.arange(139) / 125
times_ms = times * 1000

# Load TRF weights
target1 = np.load(trf_path / f'{condition[0]}_pred_data_all_{predictor}_{norm}.npz', allow_pickle=True)['target']
distr1 = np.load(trf_path / f'{condition[0]}_pred_data_all_{predictor}_{norm}.npz', allow_pickle=True)['distractor']

target2 = np.load(trf_path / f'{condition[1]}_pred_data_all_{predictor}_{norm}.npz', allow_pickle=True)['target']
distr2 = np.load(trf_path / f'{condition[1]}_pred_data_all_{predictor}_{norm}.npz', allow_pickle=True)['distractor']

# Average across conditions within each subject
target_means = np.mean(np.stack((target1, target2)), axis=0)
distr_means = np.mean(np.stack((distr1, distr2)), axis=0)

X = target_means - distr_means # subs, points, channels

ch_adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
mne.viz.plot_ch_adjacency(info, ch_adjacency, ch_names)

n_subs = 29
p = 0.05

threshold = t.ppf(1 - p/2, df=n_subs - 1)
print(threshold)

T_obs, clusters, cluster_p_vals, H0 = spatio_temporal_cluster_1samp_test(X, threshold=threshold, adjacency=ch_adjacency,
                                                                                   n_permutations=10000, tail=0, n_jobs=-1, seed=42)

ch_names = eeg_file.info['ch_names']
sig_chs = np.zeros(len(ch_names), dtype=bool)
sig_timepoints = np.zeros(len(times_ms), dtype=bool)
# clusters shape: [datapoint index], [chs index]
for (time_idx, ch_idx), pvals in zip(clusters, cluster_p_vals):
    if pvals < 0.05:
        sig_timepoints[time_idx] = True
        sig_chs[ch_idx] = True

sig_ch_names = np.array(ch_names)[sig_chs]


if len(sig_ch_names) > 0:
    target_mean = np.mean(target_means, axis=0) # average across subjects -> timepoints, ch_
    distractor_mean = np.mean(distr_means, axis=0)
    # keep only sig channels
    target_mean_chs = target_mean[:, sig_chs]
    distr_mean_chs = distractor_mean[:, sig_chs]



    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(times_ms, target_mean, color='blue', label='target')
    ax.plot(times_ms,distractor_mean, color='red', label='distractor')
else:
    print("no significant clusters detected :(")

####### compare accuracy values of target and distractor models per plane ##############
from statsmodels.stats.multitest import multipletests
acc_vals = {}
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,5))
for idx, condition in enumerate(conditions):
    if idx == 0:
        row_idx = 0
        col_idx = 0
    elif idx == 1:
        row_idx = 0
        col_idx = 1
    elif idx == 2:
        row_idx = 1
        col_idx = 0
    elif idx == 3:
        row_idx = 1
        col_idx = 1

    accuracy_path = trf_path / f"{condition}_pred_data_accuracy_vals_{norm}.npz"
    target_acc_vals = np.load(accuracy_path, allow_pickle=True)['target']
    target_acc_vals = np.mean(target_acc_vals, axis=1)
    distr_acc_vals = np.load(accuracy_path, allow_pickle=True)['distractor']
    distr_acc_vals = np.mean(distr_acc_vals, axis=1)
    # paired t-test:
    from scipy.stats import ttest_rel
    t_stat, pval = ttest_rel(target_acc_vals, distr_acc_vals) # across subs
    cohen_d = np.mean(target_acc_vals - distr_acc_vals) / np.std(target_acc_vals - distr_acc_vals)
    acc_vals[condition] = {'pval': pval, 'cohen_d': cohen_d}
    # plot accuracy values per cond:
    ax[row_idx, col_idx].boxplot([target_acc_vals, distr_acc_vals], tick_labels=['Target', 'Distractor'])
    ax[row_idx, col_idx].set_title(f'{condition}')

# avg across conds:
def extract_acc_vals(condition=''):
    accuracy_path = trf_path / f"{condition}_pred_data_accuracy_vals_{norm}.npz"
    target_acc_vals = np.load(accuracy_path, allow_pickle=True)['target']
    target_acc_vals = np.mean(target_acc_vals, axis=1)
    distr_acc_vals = np.load(accuracy_path, allow_pickle=True)['distractor']
    distr_acc_vals = np.mean(distr_acc_vals, axis=1)
    return target_acc_vals, distr_acc_vals

a1_t_vals, a1_d_vals = extract_acc_vals(condition='a1')
a2_t_vals, a2_d_vals = extract_acc_vals(condition='a2')
azimuth_t_avg = np.mean((a1_t_vals, a2_t_vals), axis=0)
azimuth_d_avg = np.mean((a1_d_vals, a2_d_vals), axis=0)
e1_t_vals, e1_d_vals = extract_acc_vals(condition='e1')
e2_t_vals, e2_d_vals = extract_acc_vals(condition='e2')
elevation_t_avg = np.mean((e1_t_vals, e2_t_vals), axis=0)
elevation_d_avg = np.mean((e1_d_vals, e2_d_vals), axis=0)

def paired_ttest(vals1, vals2):
    t_stat, pval = ttest_rel(vals1, vals2)
    cohen_d = np.mean(vals1 - vals2) / np.std(vals1 - vals2)
    return t_stat, pval, cohen_d

az_t_stat, az_p_val, az_cohen_d = paired_ttest(azimuth_t_avg, azimuth_d_avg)
el_t_stat, el_p_val, el_cohen_d = paired_ttest(elevation_t_avg, elevation_d_avg)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
vals = [[azimuth_t_avg, azimuth_d_avg], [elevation_t_avg, elevation_d_avg]]
for idx, plane in enumerate(['azimuth', 'elevation']):
    ax[idx].boxplot(vals[idx], tick_labels=['Target', 'Distractor'])
    ax[idx].set_title(f'{plane}')


# compare acc val diffs across planes
azimuth_diff = azimuth_t_avg - azimuth_d_avg
elevation_diff = elevation_t_avg - elevation_d_avg

diff_t_stat, diff_p_val, diff_cohen_d = paired_ttest(azimuth_diff, elevation_diff)

# compare within plane same streams:
az_t_tstat, az_t_pval, az_t_cohen_d = paired_ttest(a1_t_vals, a2_t_vals)
az_d_tstat, az_d_pval, az_d_cohen_d = paired_ttest(a1_d_vals, a2_d_vals)

ele_t_tstat, ele_t_pval, ele_t_cohen_d = paired_ttest(e1_t_vals, e2_t_vals)
ele_d_tstat, ele_d_pval, ele_d_cohen_d = paired_ttest(e1_d_vals, e2_d_vals)


# target stream prediction accuracy in azimuth yielded sig diffs -> need to investigate interaction
az_target_diff = a1_t_vals - a2_t_vals
az_distr_diff = a1_d_vals - a2_d_vals

interaction_t, interaction_p, interaction_d = paired_ttest(az_target_diff, az_distr_diff)

# correct for multiple comparisons -> cluster pvals of the same hypotheses family
def pval_corr(pvals_list):
    from statsmodels.stats.multitest import multipletests

    reject, pvals_corr, _, _ = multipletests(pvals_list, alpha=0.05, method='fdr_bh')
    return reject, pvals_corr

# target stream p values across planes
reject, pvals_corr = pval_corr([az_t_pval, ele_t_pval])
# hypotheses families:
# greater prediction accuracy effect in target vs distractor difference across planes - 2 pvals
# greater prediction accuracy effect in target vs distractor within planes - 4x azimuth, 4x elevation
# greater prediction accuracy effect for target model in azimuth vs elevation - 1x
# greater prediction accuracy effect for distractor model in azimuth vs elevation - 1x
# exploratory:
# no differences in prediction accuracy effect for target models within planes - 1x
# no differences in prediction accuracy effect for distractor models within planes - 1x
