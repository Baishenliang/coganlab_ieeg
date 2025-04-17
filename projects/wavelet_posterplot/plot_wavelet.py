import os
import mne
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map


HOME = os.path.expanduser("~")
Task_Tag="LexicalDecRepDelay"
save_dir=os.path.join(HOME, "Box", "CoganLab", "D_Data",Task_Tag,"Baishen_Figs","LexicalDecRepDelay")
filename = os.path.join(save_dir, 'D0096', 'wavelet', f'Auditory-tfr.h5')
filename = os.path.join(save_dir, 'D0096', 'wavelet', f'Resp-tfr.h5')

tfr=mne.time_frequency.read_tfrs(filename)
tfr.plot(picks='LFPS14',vlim=(-2,2),cmap=parula_map)
tfr.plot(picks='LFPS8',vlim=(-2,2),cmap=parula_map)
