# %% step up
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from ieeg.viz.mri import plot_on_average
from utils.group import load_stats

# %% define condition
stat_type='mask'
con='Auditory'
contrast='ave'
data=load_stats(stat_type,con,contrast)

# %% get the onsets of the activation (an effective cluster is defined as 0.2s)
spf = 1 / (times[1] - times[0])  # Calculate the sampling frequency
win_len = 0.2  # in second
win = int(win_len * spf)  # Number of samples in 0.1 seconds

onsets = {}

for ch_idx, ch_name in enumerate(chs):
    ch_data = data[ch_idx]
    found = False

    for start_idx in range(len(ch_data) - win + 1):
        win_data = ch_data[start_idx:start_idx + win]

        if np.all(win_data == 1):
            starting_time = times[start_idx]
            onsets[ch_name] = starting_time
            found = True
            break

    if not found:
        onsets[ch_name] = None  # No significant window found

# %% select channels with significant activation clusters
data_s = []
chs_s = []
chs_s_idx = []
onsets_s = []

for ch_idx, ch_name in enumerate(chs):
    if onsets[ch_name] is not None:  # Check if the channel has a valid onset
        data_s.append(data[ch_idx])  # Add the channel data to the selected data list
        chs_s.append(ch_name)        # Add the channel name to the selected channel names list
        chs_s_idx.append(ch_idx)
        onsets_s.append(onsets[ch_name])

# Convert the selected data list to a numpy array
data_s = np.array(data_s)

# %% do the ranking
sorted_indices = np.argsort(np.array(onsets_s))  # Get the indices that would sort the array
# %% rearrange the data according to sorted_indices
data_s_sorted = data_s[sorted_indices]
chs_s_sorted = [chs_s[i] for i in sorted_indices]
onsets_s_sorted = [onsets_s[i] for i in sorted_indices]

# %% plot the data

plt.figure(figsize=(2^15, 2^15))
fig, ax = plt.subplots()
ax.imshow(data_s_sorted, cmap='Reds')

ch_gap=20
time_gap=50
channel_names=chs_s_sorted[::ch_gap]
ax.set_yticks(range(0,len(channel_names)*ch_gap,ch_gap))
ax.set_yticklabels(channel_names)
time_stamps=times[::time_gap]
ax.set_xticks(range(0,len(time_stamps)*time_gap,time_gap))
ax.set_xticklabels(time_stamps)
try:
    zero_time_index = np.where(times == 0)[0][0]
    ax.axvline(x=zero_time_index, color='black', linestyle='--', linewidth=1)
except Exception as e:
    print('no zero time found')
fig.savefig('try.jpg', dpi=300)

# %% plot the significance electrodes on the average brain
elecols = [[1 - i/(len(chs_s_idx) - 1), 0, i/(len(chs_s_idx) - 1)] for i in range(len(chs_s_idx))]
elecols_s = [elecols[i] for i in sorted_indices]
fig = plot_on_average(subjs,picks=chs_s_idx,hemi='split',color=elecols_s)#, label_every=8)