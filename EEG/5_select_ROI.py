
import numpy as np
import pandas as pd
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from mtrf import model

# PARAMETERS
project_path = Path.cwd()
feature_type = 'standard'  # 'standard' or 'target_num'
model_type = 'combined' # 'envelopes', 'phonemes' as other options
n_best_channels = 12

golumbic = False
if golumbic:
    norm = 'joint'
else:
    norm = 'sep'

conditions = ['a1', 'a2', 'e1', 'e2']

# Separate input and output directories
trf_path = project_path / 'data' / 'TRF' / feature_type / model_type
output_path = trf_path / 'accuracy_based_roi'
img_path = trf_path / 'topomaps'

output_path.mkdir(parents=True, exist_ok=True)
img_path.mkdir(parents=True, exist_ok=True)

# GET CHANNEL NAMES
eeg_path = project_path / 'data' / 'sub10' / 'preprocessed'
eeg_filename = 'sub10_a1_preprocessed-raw.fif'

eeg_file = mne.io.read_raw_fif(eeg_path / eeg_filename, preload=False)
eeg_file.rename_channels({'ref': 'FCz'})

if 'AFz' in eeg_file.ch_names and len(eeg_file.ch_names) == 64:
    eeg_file.drop_channels('AFz')

all_chs = eeg_file.ch_names

print(f'Number of EEG channels: {len(all_chs)}')

# LOAD ACCURACY FROM ALL CONDITIONS
all_accuracies = []

for condition in conditions:

    filename = trf_path / f'{condition}_pred_data_accuracy_vals_{norm}.npz'
    data = np.load(filename, allow_pickle=True)

    target_r = np.asarray(data['target'])
    distractor_r = np.asarray(data['distractor'])

    print(condition, 'target:', target_r.shape, 'distractor:', distractor_r.shape)

    # Both target and distractor contribute to channel selection
    all_accuracies.append(target_r)
    all_accuracies.append(distractor_r)

# COMBINE EVERYTHING
# Shape: conditions x streams x subjects x channels, flattened to 8 x subjects x channels
all_accuracies = np.stack(all_accuracies, axis=0)

print('\nCombined shape:')
print(all_accuracies.shape)

# GRAND-AVERAGE ACCURACY
grand_accuracy = np.mean(all_accuracies, axis=(0, 1))

print('\nGrand-average accuracy shape:')
print(grand_accuracy.shape)

assert len(grand_accuracy) == len(all_chs), (f'Accuracy has {len(grand_accuracy)} channels but EEG channel list has {len(all_chs)}.')

# RANK CHANNELS
ranking_idx = np.argsort(grand_accuracy)[::-1]

ranking = pd.DataFrame({'Rank': np.arange(1, len(all_chs) + 1), 'Channel_index': ranking_idx, 'Channel': [all_chs[i] for i in ranking_idx], 'Mean_prediction_accuracy': grand_accuracy[ranking_idx]})

print('\nCHANNEL RANKING:')
print(ranking.to_string(index=False))

# SELECT TOP N
best_idx = ranking_idx[:n_best_channels]
best_channels = [all_chs[i] for i in best_idx]

print(f'\nTOP {n_best_channels} CHANNELS:')
print(best_channels)

print('\nIndices:')
print(best_idx.tolist())

# SAVE ROI
ranking.to_csv(output_path / f'channel_accuracy_ranking_{norm}.csv', index=False)

np.savez(output_path / f'top_{n_best_channels}_channels_{norm}.npz', channels=np.array(best_channels), indices=best_idx, accuracy=grand_accuracy[best_idx])

# PREPARE TOPOGRAPHIC INFORMATION
topo_chs = ['FCz' if ch == 'ref' else ch for ch in all_chs]

info = mne.create_info(ch_names=topo_chs, sfreq=125, ch_types='eeg')
info.set_montage('standard_1020', on_missing='ignore')

# PLOT GRAND-AVERAGE ACCURACY
fig, ax = plt.subplots(figsize=(8, 7))

im, cn = mne.viz.plot_topomap(grand_accuracy, info, axes=ax, show=False, contours=6, names=topo_chs)

fig.colorbar(im, ax=ax, label='Prediction accuracy (r)')
ax.set_title('Grand-average cross-validated prediction accuracy')

fig.savefig(img_path / f'grand_average_topomap_accuracy_vals_{norm}.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

# INVESTIGATE EACH SUBCONDITION SEPARATELY
subcondition_accuracies = {}

for condition in conditions:

    filename = trf_path / f'{condition}_pred_data_accuracy_vals_{norm}.npz'
    data = np.load(filename, allow_pickle=True)

    target_r = np.asarray(data['target'])
    distractor_r = np.asarray(data['distractor'])

    # Shape: 2 x subjects x channels
    accuracies = np.stack((target_r, distractor_r), axis=0)

    # Average across stream and subject
    condition_accuracy = np.mean(accuracies, axis=(0, 1))
    subcondition_accuracies[condition] = condition_accuracy

    # RANK CHANNELS
    ranking_idx = np.argsort(condition_accuracy)[::-1]

    ranking = pd.DataFrame({'Rank': np.arange(1, len(all_chs) + 1), 'Channel_index': ranking_idx, 'Channel': [all_chs[i] for i in ranking_idx], 'Mean_prediction_accuracy': condition_accuracy[ranking_idx]})

    print(f'\n{condition.upper()} CHANNEL RANKING:')
    print(ranking.to_string(index=False))

    ranking.to_csv(output_path / f'{condition}_channel_accuracy_ranking_{norm}.csv', index=False)

    # PLOT TOPOMAP
    fig, ax = plt.subplots(figsize=(8, 7))

    im, cn = mne.viz.plot_topomap(condition_accuracy, info, axes=ax, show=False, contours=6, names=topo_chs)

    fig.colorbar(im, ax=ax, label='Prediction accuracy (r)')
    ax.set_title(f'{condition.upper()} grand-average cross-validated prediction accuracy')

    fig.savefig(img_path / f'{condition}_topomaps_accuracy_vals_{norm}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

print(f'\nFinished channel ranking for: {feature_type}')
print(f'Rankings saved to: {output_path}')
print(f'Topomaps saved to: {img_path}')
print(f'Normalisation: {norm}')



# ============================================================
# CHANNEL RANK CONSISTENCY ACROSS SUBJECTS, CONDITIONS AND STREAMS
# ============================================================

rank_summaries = []

for condition in conditions:

    filename = trf_path / f'{condition}_pred_data_accuracy_vals_{norm}.npz'

    with np.load(filename, allow_pickle=True) as data:
        stream_data = {'target': np.asarray(data['target']), 'distractor': np.asarray(data['distractor'])}

    for stream, accuracy in stream_data.items():

        assert accuracy.ndim == 2
        assert accuracy.shape[1] == len(all_chs)

        # Rank channels within each subject (1 = highest accuracy)
        subject_ranks = np.argsort(np.argsort(-accuracy, axis=1), axis=1) + 1

        summary = pd.DataFrame({
            'Condition': condition,
            'Stream': stream,
            'Channel': all_chs,
            'Channel_index': np.arange(len(all_chs)),
            'Mean_accuracy': np.mean(accuracy, axis=0),
            'Median_rank': np.median(subject_ranks, axis=0),
            'Rank_Q25': np.percentile(subject_ranks, 25, axis=0),
            'Rank_Q75': np.percentile(subject_ranks, 75, axis=0),
            'Top5_percent': 100 * np.mean(subject_ranks <= 5, axis=0),
            'Top10_percent': 100 * np.mean(subject_ranks <= 10, axis=0)})

        rank_summaries.append(summary)

# All channels, separately for each condition and stream
rank_consistency = pd.concat(rank_summaries, ignore_index=True)
rank_consistency.to_csv(output_path / f'channel_rank_consistency_by_condition_stream_{norm}.csv', index=False)

# Overall consistency across all 8 condition-stream combinations
overall_consistency = rank_consistency.groupby('Channel').agg(
    Mean_median_rank=('Median_rank', 'mean'),
    Worst_median_rank=('Median_rank', 'max'),
    Mean_top10_percent=('Top10_percent', 'mean'),
    Minimum_top10_percent=('Top10_percent', 'min'),
    Mean_accuracy=('Mean_accuracy', 'mean')).reset_index()

overall_consistency = overall_consistency.sort_values(['Worst_median_rank', 'Mean_median_rank', 'Mean_top10_percent'],
    ascending=[True, True, False]).reset_index(drop=True)

overall_consistency.insert(0, 'Overall_rank', np.arange(1, len(overall_consistency) + 1))
overall_consistency.to_csv(output_path / f'overall_channel_rank_consistency_{norm}.csv', index=False)

print('\nTOP 20 MOST CONSISTENT CHANNELS:')
print(overall_consistency.head(20).to_string(index=False))

# Automatically select top N channels based on consistency
best_channels = overall_consistency['Channel'].head(n_best_channels).tolist()

plot_data = rank_consistency[rank_consistency['Channel'].isin(best_channels)].copy()
plot_data['Condition_stream'] = plot_data['Condition'] + '_' + plot_data['Stream']

comparison = plot_data.pivot(index='Channel', columns='Condition_stream', values='Median_rank')
comparison = comparison.reindex(index=best_channels, columns=[f'{c}_{s}' for c in conditions for s in ['target', 'distractor']])

print(f'\nTOP {n_best_channels} CHANNELS BY RANK CONSISTENCY:')
print(comparison.to_string())

comparison.to_csv(output_path / f'top_channels_rank_consistency_{norm}.csv')

# Plot
fig, ax = plt.subplots(figsize=(11, 5))
comparison.T.plot(marker='o', ax=ax)

ax.set_xlabel('Condition and stream')
ax.set_ylabel('Median rank across subjects')
ax.set_title('Channel rank consistency across conditions and streams')
ax.invert_yaxis()
ax.grid(alpha=0.3)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Channel', bbox_to_anchor=(1.02, 1), loc='upper left')

fig.tight_layout()
fig.savefig(img_path / 'channel_rank_consistency.pdf', dpi=300, bbox_inches='tight')

plt.show()
plt.close(fig)
