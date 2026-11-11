'''
1. load data, downsample, apply montage
3. concatenate
4. bandpass 1-8Hz
5. select bad channels
6. apply ICA
7. interpolate
8. average reference
9. save preprocessed eeg file
'''

import os
from pathlib import Path
import copy
import mne
from meegkit import dss
import json

# Define parameters
i = 1
conditions = ['a1', 'a2', 'e1', 'e2']
condition = conditions[i]
subject = 'sub33'

project_path = Path.cwd()
eeg_path = project_path / 'data' / subject / 'eeg' # adjust later accordingly
save_path = project_path / 'data' / subject / 'preprocessed'
electrodes_path = project_path / 'data' / 'misc' / 'electrode_names.json'
os.makedirs(save_path, exist_ok=True)

# import electrodes_names:
with open(electrodes_path) as json_file:
    electrodes = json.load(json_file)

# 1. load raw EEG data, downsample and apply montage
raw_list = []
for eeg_files in eeg_path.iterdir():
    if condition in eeg_files.name:
        if 'vhdr' in eeg_files.name:
            if subject == 'sub33' and 'a2_3' in eeg_files.name:
                continue
            elif subject == 'sub35' and 'a1_0' in eeg_files.name:
                continue
            else:
                eeg_data = mne.io.read_raw_brainvision(eeg_files, preload=True)
                eeg_data.resample(sfreq=125) # downsample to 125 Hz
                if len(eeg_data.info['ch_names']) == 67:
                    print('Data was recorded with the VERY OLD acticap set. Removing unwanted channels...')
                    eeg_data.rename_channels(electrodes)
                    eeg_data.drop_channels(['A1', 'A2', 'M2', 'PO9', 'PO10'])
                    eeg_data.set_montage('standard_1020')
                else:
                    print('Data was recorded with the new acticap set. Remving unwanted channel...')
                    eeg_data.set_montage('standard_1020')
                    eeg_data.drop_channels(['Iz', 'AFz'])
            raw_list.append(eeg_data)

# 2. concatenate EEG data:
concat_data = mne.concatenate_raws(raw_list)

# 3. bandpass filter 1-8 Hz
eeg_filter = copy.deepcopy(concat_data)
data = mne.io.RawArray(data=concat_data.get_data(), info=concat_data.info)
eeg_notch, iterations = dss.dss_line(concat_data.get_data().T, fline=50, sfreq=data.info["sfreq"], nfft=400)

eeg_filter._data = eeg_notch.T
eeg_filtered = copy.deepcopy(eeg_filter).filter(1, 8)

# select bad channels and shit segments with global noise
eeg_filtered.plot()

eeg_filtered.plot_psd()

# Apply ICA
ica_eeg = copy.deepcopy(eeg_filtered)
ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=99)
ica.fit(ica_eeg)

ica.plot_components()

ica.plot_sources(ica_eeg)

ica.apply(ica_eeg)

# interpolate bad channels
eeg_interp = copy.deepcopy(ica_eeg)
eeg_interp.interpolate_bads()

# use avg ref
eeg_avg = copy.deepcopy(eeg_interp)
eeg_avg.add_reference_channels('ref')
eeg_avg.set_eeg_reference(ref_channels='average')

# check bad segments and remove marking if gone
eeg_avg.plot()

# save
eeg_avg.save(save_path/f'{subject}_{condition}_preprocessed-raw.fif', overwrite=True)


