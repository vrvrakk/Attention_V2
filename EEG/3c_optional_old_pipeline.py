
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(r"C:\Windows\Fonts\times.ttf")
plt.rcParams['font.family'] = 'Times New Roman'
import mne
from mne.stats import permutation_cluster_test
from statsmodels.stats.multitest import multipletests
from scipy.stats import t

# Parameters
plane = 'azimuth'

predictor = 'phonemes'

model_type = 'combined' # envelopes, phonemes as other options

norm = 'sep'

# Select feature type: 'standard' or 'target_num'
feature_type = 'standard'


project_path = Path.cwd()

# Separate input/output folders for each feature type
trf_path = project_path / 'data' / 'TRF' / feature_type / model_type / 'concatenated'
fig_path = project_path / 'data' / 'figures' / feature_type / model_type / 'concatenated'
fig_path.mkdir(parents=True, exist_ok=True)

# Load channel names
eeg_path = project_path / 'data' / 'sub10' / 'preprocessed'
eeg_filename = 'sub10_a1_preprocessed-raw.fif'
eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=False)

# Rename reconstructed reference channel to FCz for ROI selection
eeg_file.rename_channels({'ref': 'FCz'})
all_chs = eeg_file.ch_names

# Select ROI
if predictor == 'envelopes':
    sel_ch = ['FCz']
else:
    sel_ch = ['F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8']

ch_idx = np.isin(all_chs, sel_ch)

assert ch_idx.sum() == len(sel_ch), f'Missing ROI channels: {set(sel_ch) - set(all_chs)}'

# Time vector
# times = np.arange(64) / 125
# times_ms = times * 1000
times_ms = [-0.104, -0.096, -0.088, -0.08 , -0.072, -0.064, -0.056, -0.048,
       -0.04 , -0.032, -0.024, -0.016, -0.008,  0.   ,  0.008,  0.016,
        0.024,  0.032,  0.04 ,  0.048,  0.056,  0.064,  0.072,  0.08 ,
        0.088,  0.096,  0.104,  0.112,  0.12 ,  0.128,  0.136,  0.144,
        0.152,  0.16 ,  0.168,  0.176,  0.184,  0.192,  0.2  ,  0.208,
        0.216,  0.224,  0.232,  0.24 ,  0.248,  0.256,  0.264,  0.272,
        0.28 ,  0.288,  0.296,  0.304,  0.312,  0.32 ,  0.328,  0.336,
        0.344,  0.352,  0.36 ,  0.368,  0.376,  0.384,  0.392,  0.4  ,
        0.408,  0.416,  0.424,  0.432,  0.44 ,  0.448,  0.456,  0.464,
        0.472,  0.48 ,  0.488,  0.496,  0.504,  0.512,  0.52 ,  0.528,
        0.536,  0.544,  0.552,  0.56 ,  0.568,  0.576,  0.584,  0.592,
        0.6  ,  0.608,  0.616,  0.624,  0.632,  0.64 ,  0.648,  0.656,
        0.664,  0.672,  0.68 ,  0.688,  0.696,  0.704,  0.712,  0.72 ,
        0.728,  0.736,  0.744,  0.752,  0.76 ,  0.768,  0.776,  0.784,
        0.792,  0.8  ,  0.808,  0.816,  0.824,  0.832,  0.84 ,  0.848,
        0.856,  0.864,  0.872,  0.88 ,  0.888,  0.896,  0.904,  0.912,
        0.92 ,  0.928,  0.936,  0.944,  0.952,  0.96 ,  0.968,  0.976,
        0.984,  0.992,  1.   ]
times_ms = np.asarray(times_ms) * 1000

TIME_WINDOWS = [(50, 150), (150, 250), (250, 350), (350, 500)]

from mne.stats import permutation_cluster_test, fdr_correction

def run_windowed_permutations(target_mean, distr_mean):

    all_clusters = []
    all_pvals = []

    for w_start, w_end in TIME_WINDOWS:

        idx = np.where((times_ms >= w_start) & (times_ms < w_end))[0]

        target_window = target_mean[:, idx]
        distractor_window = distr_mean[:, idx]

        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([target_window, distractor_window], threshold=None,
            n_permutations=1000, n_jobs=-1, seed=42)

        for cl, pval in zip(clusters, cluster_p_values):
            cluster_idx = idx[cl[0]]

            all_clusters.append({'window': (w_start, w_end), 'indices': cluster_idx, 'p_uncorrected': pval})

            all_pvals.append(pval)

    if not all_pvals:
        return []

    # FDR correction across clusters from ALL four windows
    reject, pvals_fdr = fdr_correction(np.asarray(all_pvals), alpha=0.05)

    significant_clusters = []

    for cluster, significant, p_fdr in zip(
        all_clusters, reject, pvals_fdr):
        if significant:
            cluster['p_corrected'] = p_fdr
            significant_clusters.append(cluster)

    return significant_clusters


# Load TRF weights
target = np.load(trf_path / f'{plane}_pred_data_all_{predictor}_{norm}.npz', allow_pickle=True)['target']
distr = np.load(trf_path / f'{plane}_pred_data_all_{predictor}_{norm}.npz', allow_pickle=True)['distractor']


def get_roi_data(target_means, distr_means, predictor=''):

    # Select ROI and average channels within each subject
    target_mean = np.mean(target_means[:, :, ch_idx], axis=-1)
    distractor_mean = np.mean(distr_means[:, :, ch_idx], axis=-1)

    # SEM across subjects, after ROI averaging
    target_sem = np.std(target_mean, axis=0, ddof=1) / np.sqrt(target_mean.shape[0])
    distractor_sem = np.std(distractor_mean, axis=0, ddof=1) / np.sqrt(distractor_mean.shape[0])

    # Save subject-level ROI waves
    np.savez(trf_path / f'{plane}_mean_trf_{predictor}_{norm}.npz', target_mean=target_mean, distractor_mean=distractor_mean)

    return target_mean, distractor_mean, target_sem, distractor_sem


target_mean, distr_mean, t_sem, d_sem = get_roi_data(target, distr, predictor=predictor)


def plot_cluster_topomaps(target, distr, all_times):

    # Prepare channel information for MNE topomaps
    info = eeg_file.info.copy()

    if 'AFz' in info['ch_names']:
        info = mne.pick_info(info, mne.pick_channels(info['ch_names'], include=[ch for ch in info['ch_names'] if ch != 'AFz'], ordered=True))

    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')

    # Subject-level attended - unattended TRF differences
    differences = target - distr

    for cluster_idx, (t_start, t_end) in enumerate(all_times, start=1):

        # Convert significant time window to sample indices
        tidx = np.where((times_ms >= t_start) & (times_ms <= t_end))[0]

        # Average across time, then subjects
        subject_maps = np.mean(differences[:, tidx, :], axis=1) # original data subs, time, chs
        mean_map = np.mean(subject_maps, axis=0)

        # Symmetric limits for signed differences
        vmax = np.max(np.abs(mean_map))

        fig, ax = plt.subplots(figsize=(5, 4))

        im, _ = mne.viz.plot_topomap(mean_map, info, axes=ax, show=False, cmap='magma', vlim=(-vmax, vmax), contours=0)

        ax.set_title(f'{plane.capitalize()} | {predictor}\n{t_start:.0f}–{t_end:.0f} ms')

        cbar = plt.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label('Attended − Unattended TRF weight')

        fig.tight_layout()

        filename = f'{plane}_{predictor}_cluster_{cluster_idx}_topomap_{norm}.pdf'
        fig.savefig(fig_path / filename, dpi=600, bbox_inches='tight')
        plt.close(fig)

        print(f'Saved topomap: {filename}')



def plot_weights(target_mean, distr_mean, target_sem, distractor_sem):

    fig, ax = plt.subplots(figsize=(10, 5))

    # Grand-average TRFs
    t_mean = np.mean(target_mean, axis=0)
    d_mean = np.mean(distr_mean, axis=0)

    ax.plot(times_ms, t_mean, 'b-', linewidth=2, label='Target')
    ax.fill_between(times_ms, t_mean - target_sem, t_mean + target_sem, color='b', alpha=0.3)

    ax.plot(times_ms, d_mean, 'r-', linewidth=2, label='Distractor')
    ax.fill_between(times_ms, d_mean - distractor_sem, d_mean + distractor_sem, color='r', alpha=0.3)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('TRF weight', fontsize=18)

    # Windowed cluster permutation tests + FDR correction
    significant_clusters = run_windowed_permutations(target_mean, distr_mean)

    all_pvals = []
    all_times = []

    for cluster in significant_clusters:

        idx = cluster['indices']

        t_start = times_ms[idx[0]]
        t_end = times_ms[idx[-1]]

        pval = cluster['p_corrected']

        all_pvals.append(pval)
        all_times.append([t_start, t_end])

        print(f'{plane}: significant cluster '
            f'{t_start:.1f} - {t_end:.1f} ms, '
            f'Bonferroni-corrected p={pval:.4f}')

        # Shade only the actual significant cluster
        ax.axvspan(t_start, t_end, color='gray', alpha=0.2)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlim([0., 500.])

    ax.legend()
    fig.tight_layout()

    filename = f'{plane}_{predictor}_trf_weights_{norm}.pdf'
    fig.savefig(fig_path / filename, dpi=600, bbox_inches='tight')

    if all_times:
        plot_cluster_topomaps(target, distr, all_times)
    else:
        print('No significant clusters found. No topomaps generated.')

    return all_pvals, all_times



plot_weights(target_mean, distr_mean, t_sem, d_sem)


print(f'\nFinished: {plane} | {predictor} | {feature_type} | {model_type}| {norm}')
print(f'Figures saved to: {fig_path}')

