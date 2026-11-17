import os
from pathlib import Path
from collections import Counter
import re
import mne
import pandas as pd

# Define parameters
i = 3
conditions = ['a1', 'a2', 'e1', 'e2']
condition = conditions[i]

subjects = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19',
    'sub20', 'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27',
    'sub28', 'sub29', 'sub30', 'sub31', 'sub32', 'sub33', 'sub34', 'sub35',
    'sub36', 'sub37', 'sub38', 'sub39', 'sub40']

project_path = Path.cwd()
audit_path = project_path / 'data' / 'event_audit' / condition
os.makedirs(audit_path, exist_ok=True)

summary_rows = []
event_rows = []
missing_rows = []

stream1_codes = set(range(1, 10))
stream2_codes = set(range(65, 74))
press_codes = set(range(129, 138))
known_codes = stream1_codes | stream2_codes | press_codes | {64}

target_stream = 's1' if condition in ['a1', 'e1'] else 's2'
target_codes = stream1_codes if target_stream == 's1' else stream2_codes
axis = 'azimuth' if condition.startswith('a') else 'elevation'

for subject in subjects:
    eeg_path = project_path / 'data' / subject / 'eeg'
    csv_path = project_path / 'data' / subject / 'csv' / f'{subject}.csv'

    blocks_info = pd.read_csv(csv_path)

    # Select metadata for the condition
    final_blocks = blocks_info[(blocks_info['block_seq'] == target_stream) & (blocks_info['block_condition'] == axis)]
    # Exclude unusable blocks
    if condition == 'a1' and subject == 'sub35':
        final_blocks = final_blocks.iloc[1:]

    if condition == 'a2' and subject == 'sub33':
        final_blocks = final_blocks.drop(final_blocks.index[3])
    # Locate files in block order
    marker_files = sorted([f for f in eeg_path.iterdir() if f.suffix.lower() == '.vhdr' and re.search(rf'{condition}_(\d+)$', f.stem)],
        key=lambda f: int(re.search(r'_(\d+)$', f.stem).group(1)))

    if condition == 'a1' and subject == 'sub35':
        marker_files = [f for f in marker_files if not re.search(r'a1_0$', f.stem)]

    if condition == 'a2' and subject == 'sub33':
        marker_files = [f for f in marker_files if not re.search(r'a2_3$', f.stem)]

    if len(marker_files) != len(final_blocks):
        print(f'WARNING: {subject}: {len(marker_files)} files, {len(final_blocks)} metadata blocks')

    for block_index, eeg_file in enumerate(marker_files):
        block = final_blocks.iloc[block_index]
        original_block_index = int(re.search(r'_(\d+)$', eeg_file.stem).group(1))

        if block_index >= len(final_blocks):
            missing_rows.append({'subject': subject, 'condition': condition, 'block_index': block_index, 'reason': 'No corresponding metadata block'})
            continue

        block = final_blocks.iloc[block_index]

        # Read annotations without loading EEG signal
        eeg = mne.io.read_raw_brainvision(eeg_file, preload=False, verbose=False)

        # Preserve original trigger codes
        event_id = {}
        for description in eeg.annotations.description:
            match = re.search(r'S\s*(\d+)', description)
            if match:
                event_id[description] = int(match.group(1))

        events, ids = mne.events_from_annotations(eeg, event_id=event_id, verbose=False)
        counts = Counter(events[:, 2])

        expected = int(block['Target Number'])
        expected_stim = expected if target_stream == 's1' else expected + 64
        expected_press = expected + 128

        # Dominant target-stream stimulus
        observed_stim = max(target_codes, key=lambda code: counts[code])
        observed_target = observed_stim if target_stream == 's1' else observed_stim - 64

        # Dominant button press
        observed_press = max(press_codes, key=lambda code: counts[code])
        observed_press_target = observed_press - 128 if counts[observed_press] > 0 else None

        # Unexplained codes
        random_events = {int(code): int(count) for code, count in counts.items() if code not in known_codes}

        summary_rows.append({
            'subject': subject, 'condition': condition, 'block_index': block_index,
            'marker_file': eeg_file.name,
            'expected_target': expected, 'observed_target': observed_target,
            'target_match': expected == observed_target,
            'observed_press_target': observed_press_target,
            'press_target_match': expected == observed_press_target,
            'stim_press_match': observed_target == observed_press_target,
            'expected_stim_count': counts[expected_stim],
            'expected_press_count': counts[expected_press],
            'observed_stim_count': counts[observed_stim],
            'observed_press_count': counts[observed_press],
            'invalid_press_135_count': counts[135],
            'random_event_count': sum(random_events.values()),
            'random_event_codes': str(random_events)})

        # Counts of ALL events
        for code, count in sorted(counts.items()):
            event_rows.append({
                'subject': subject, 'condition': condition, 'block_index': block_index, 'marker_file': eeg_file.name,
                'event_code': int(code), 'count': int(count),
                'category': 's1' if code in stream1_codes else 's2' if code in stream2_codes else 'button_press' if code in press_codes else 'block_marker' if code == 64 else 'unexplained'})

summary = pd.DataFrame(summary_rows)
event_counts = pd.DataFrame(event_rows)
missing = pd.DataFrame(missing_rows)

summary.to_csv(audit_path / 'block_summary.csv', index=False)
event_counts.to_csv(audit_path / 'event_counts.csv', index=False)
missing.to_csv(audit_path / 'missing_files.csv', index=False)

print(summary.to_string(index=False))
print(f'\nSaved audit to: {audit_path}')