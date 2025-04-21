#%% load mne epochs
import os
import mne

HOME = os.path.expanduser("~")
Task_Tag="LexicalDecRepDelay"

save_dir=os.path.join(HOME, "Box", "Coganlab\\BIDS-1.0_LexicalDecRepDelay\\BIDS\\derivatives\\stats")
filename = os.path.join(save_dir, 'D0096','Auditory_inRep_power-epo.fif')

epochs=mne.read_epochs(filename)

#%% generate new epoch data:
# should be: shape (n_epochs, n_channels, n_times)
import numpy as np
data=epochs.get_data()
data_new=np.random.rand(np.shape(data)[0],np.shape(data)[1],np.shape(data)[2])

#%% generate new epoch
new_epochs = mne.EpochsArray(data_new, epochs.info,tmin=epochs.tmin)