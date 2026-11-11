'''
1. load concatenated files + json helper files
2. load csv - extract info for each block - voice, target num, stream order
3. extract events from concatenated data - define start and end of each block
4. create an np zeros array
5. enter 1s whenever there was a button press - response array - motor noise tracking
6. loop through data event array - for each segment, match corresponding number stimulus of correct voice
7. do this for stream 1 and 2 respectively
8. save response arrays, stimuli arrays
'''
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne
import textgrids

# define params and paths
i = 3
conditions = ['a1', 'a2', 'e1', 'e2']
condition = conditions[i]

if condition in ['a1', 'a2']:
    plane = 'azimuth'
else:
    plane = 'elevation'

if condition in ['a1', 'e1']:
    target_stream = 's1'
else:
    target_stream = 's2'

subjects = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19',
    'sub20', 'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27',
    'sub28', 'sub29', 'sub30', 'sub31', 'sub32', 'sub33', 'sub34', 'sub35',
    'sub36', 'sub37', 'sub38', 'sub39', 'sub40'] # sub33 and sub35 - a1_0 and a2_3 header and marker files missing...


for subject in subjects:

    project_path = Path.cwd()
    concat_path = project_path / 'data' / subject / 'preprocessed'
    csv_path = project_path / 'data' / subject / 'csv'
    voices_path = project_path / 'data' / 'voices_english' / 'downsampled'
    mfa_path = project_path / 'data' / 'MFA'

    # load concatenated eeg data:
    concat_data = mne.io.read_raw(concat_path/f'{subject}_{condition}_preprocessed-raw.fif', preload=True)
    # get events and event ids:
    events, ids = mne.events_from_annotations(concat_data)

    with open(csv_path/f'{subject}.csv') as f:
        blocks_info = pd.read_csv(f)

    # keep only relevant blocks to the condition selected
    condition_blocks = blocks_info[blocks_info['block_condition'] == plane]
    final_blocks = condition_blocks[condition_blocks['block_seq'] == target_stream]
    voices_order = list(final_blocks['Voices'])
    target_nums = list(final_blocks['Target Number'])
    if condition == 'a1' and subject == 'sub35':
        voices_order = voices_order[1:]
        final_blocks = final_blocks.iloc[1:]
        target_nums = target_nums[1:]

    if condition == 'a2' and subject == 'sub33':
        voices_order.pop(3)
        target_nums.pop(3)
        final_blocks = final_blocks.drop(final_blocks.index[3])


    # load events mapping:
    helper_path = project_path / 'data' / 'misc'

    with open(helper_path / 'eeg_events.json', 'r') as f:
        events_mapping = json.load(f)

    # if key from ids is in events_dict - keep those events from event array
    stream1_nums = []
    stream2_nums = []
    response_nums = []



    for key, value in ids.items():
        if key in events_mapping['s1_events']:
            num = ids[key]
            stream1_nums.append(num)
        elif key in events_mapping['s2_events']:
            num = ids[key]
            stream2_nums.append(num)
        elif key in events_mapping['response_events']:
            num = ids[key]
            response_nums.append(num)

    # get separate event arrays for s1 and s2 and responses:
    s1_events = []
    s2_events = []
    response_events = []

    all_events = []

    all_nums = np.concatenate((stream1_nums, stream2_nums, response_nums))

    for event in events:
        if event[-1] in stream1_nums:
            s1_events.append(event)
            all_events.append(event)
        elif event[-1] in stream2_nums:
            s2_events.append(event)
            all_events.append(event)
        elif event[-1] in response_nums:
            response_events.append(event)
            all_events.append(event)

    # numpy stack
    s1_events = np.stack(s1_events)
    s2_events = np.stack(s2_events)
    response_events = np.stack(response_events)


    # get start and end timepoints of blocks from events:
    segment_stop = ids['Stimulus/S 64']

    segment_timepoints = []
    i = -1
    for event in events:
        if event[-1] == segment_stop:
            i += 1
            block_end = event[0]
            segment_timepoints.append(event[0])
        else:
            continue
    total_blocks = len(segment_timepoints)

    def event_arrays(s_events):
        # np zero array aligned with concatenated eeg data:
        empty_arr = np.zeros(concat_data.get_data().shape[-1])

        s_arr = empty_arr.copy()

        event_timepoints = s_events[:, 0]
        for idx, value in enumerate(s_arr):
            if idx in event_timepoints:
                event_num = s_events[s_events[:, 0] == idx, -1][0]
                s_arr[idx] = event_num
            elif idx in segment_timepoints:
                s_arr[idx] = 9999
            else:
                continue
        return s_arr

    s1_arr = event_arrays(s1_events)
    s1_arr = s1_arr.astype(int)
    s2_arr = event_arrays(s2_events)
    s2_arr = s2_arr.astype(int)

    s1_ids = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3', 'Stimulus/S  4', 'Stimulus/S  5', 'Stimulus/S  6','Stimulus/S  8', 'Stimulus/S  9']
    s2_ids = ['Stimulus/S 65', 'Stimulus/S 66','Stimulus/S 67','Stimulus/S 68','Stimulus/S 69','Stimulus/S 70','Stimulus/S 72','Stimulus/S 73']
    s1_nums = [num for key, num in ids.items() if key in s1_ids]
    s2_nums = [num for key, num in ids.items() if key in s2_ids]
    actual_nums = [1, 2, 3, 4, 5, 6, 8, 9]
    # assign correct numbers to s_arr...
    def assign_corr_vals(s_arr, s_nums):
        for arr_idx, value in enumerate(s_arr):
            if value in s_nums:
                value_idx = s_nums.index(value)
                actual_value = actual_nums[value_idx]
                s_arr[arr_idx] = actual_value
        return s_arr

    s1_arr_corr = assign_corr_vals(s1_arr, s1_nums)
    s2_arr_corr = assign_corr_vals(s2_arr, s2_nums)

    response_arr = event_arrays(response_events)
    response_arr = response_arr.astype(int)

    # convert responses to 1:
    for idx, resp in enumerate(response_arr):
        if resp not in [0, 9999]:
            response_arr[idx] = 1
        else:
            continue

    # sort s1 and s2 envelope arrays based on segment and voice of each seg
    def get_features(s_arr):
        env_arr = np.zeros(s_arr.shape)
        phonemes_arr = np.zeros(s_arr.shape)
        i = 0
        while i < total_blocks:
            for idx, number in enumerate(s_arr):
                block_end_timepoint = segment_timepoints[i]
                voice = voices_order[i]
                if idx <= block_end_timepoint:
                    if number not in  [0, 9999]:
                        print(f'Number {number}, currently in block {i}, with voice {voice}. /n'
                              f'Extracting relevant wav files...')
                        if number < 7:
                            wav_idx = number - 1
                            mfa_idx = number
                        else:
                            wav_idx = number - 2
                            mfa_idx = number - 1
                        # go into respective folder and extract envelope:
                        voice_folder = voices_path / voice
                        wav_file = list(voice_folder.iterdir())[wav_idx]
                        envelope = np.load(wav_file)
                        # insert env arr into env_arr:
                        env_start = idx
                        env_end = min(idx + len(envelope), len(env_arr))
                        env_arr[env_start:env_end] = envelope
                        # now phonemes
                        mfa_voice_path = mfa_path / f'aligned_{voice}'
                        mfa_voice = list(mfa_voice_path.iterdir())[mfa_idx]
                        print(f'For Number: {number}, mfa file is : {mfa_voice}, and envelope is: {wav_file}')
                        mfa_file = textgrids.TextGrid(mfa_voice)
                        phones = mfa_file['phones']
                        first_phoneme = True
                        for phonemes in phones:
                            if not phonemes.text:
                                continue
                            else:
                                if first_phoneme:
                                    first_phoneme = False
                                    continue
                                phoneme_start = int(idx + (phonemes.xmin * 125))
                                # phoneme_end = int(min(phoneme_start + (phonemes.dur * 125),len(phonemes_arr)))
                                phonemes_arr[phoneme_start] = 1
                    else:
                        continue
                else:
                    i += 1
                    if i == total_blocks:
                        break
        return env_arr, phonemes_arr

    env1_arr, phonemes1_arr = get_features(s1_arr)
    env2_arr, phonemes2_arr = get_features(s2_arr)

    def get_target_num_features(s_arr):
        env_arr = np.zeros(s_arr.shape)
        phonemes_arr = np.zeros(s_arr.shape)
        i = 0
        while i < total_blocks:
            for idx, number in enumerate(s_arr):
                block_end_timepoint = segment_timepoints[i]
                voice = voices_order[i]
                target_num = target_nums[i]
                if idx <= block_end_timepoint:
                    if number == target_num:
                        if number < 7:
                            wav_idx = number - 1
                            mfa_idx = number
                        else:
                            wav_idx = number - 2
                            mfa_idx = number - 1
                        # go into respective folder and extract envelope:
                        voice_folder = voices_path / voice
                        wav_file = list(voice_folder.iterdir())[wav_idx]
                        envelope = np.load(wav_file)
                        # insert env arr into env_arr:
                        env_start = idx
                        env_end = min(idx + len(envelope), len(env_arr))
                        env_arr[env_start:env_end] = envelope
                        mfa_voice_path = mfa_path / f'aligned_{voice}'
                        mfa_voice = list(mfa_voice_path.iterdir())[mfa_idx]
                        print(f'For Number: {number}, mfa file is : {mfa_voice}, and envelope is: {wav_file}')
                        mfa_file = textgrids.TextGrid(mfa_voice)
                        phones = mfa_file['phones']
                        first_phoneme = True
                        for phonemes in phones:
                            if not phonemes.text:
                                continue
                            else:
                                if first_phoneme:
                                    first_phoneme = False
                                    continue
                                phoneme_onset = int(idx + (phonemes.xmin * 125))
                                phonemes_arr[phoneme_onset] = 1
                    else:
                        continue
                else:
                    i += 1
                    if i == total_blocks:
                        break
        return env_arr, phonemes_arr

    target_num_env1_arr, target_num_phonemes1_arr = get_target_num_features(s1_arr)
    target_num_env2_arr, target_num_phonemes2_arr = get_target_num_features(s2_arr)

    # save envelope arrays and response arrays:
    save_path = project_path / 'data' / subject / 'features'
    os.makedirs(save_path, exist_ok=True)


    np.savez(save_path/f'{subject}_{condition}_features.npz',
             env1=env1_arr, env2=env2_arr, phon1=phonemes1_arr, phon2=phonemes2_arr, responses=response_arr)

    np.savez(save_path/f'{subject}_{condition}_target_num_features.npz',
             env1=target_num_env1_arr, env2=target_num_env2_arr,
             phon1=target_num_phonemes1_arr, phon2=target_num_phonemes2_arr, responses=response_arr)


