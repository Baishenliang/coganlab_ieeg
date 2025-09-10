import os
import mne
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12

HOME = os.path.expanduser("~")
Task_Tag="LexicalDecRepDelay"
save_dir=os.path.join(HOME, "Box", "CoganLab", "D_Data",Task_Tag,"Baishen_Figs","LexicalDecRepDelay")

for fname,size in zip(
        ('Auditory-tfr.h5','Resp-tfr.h5'),
        ((10, 2),(4, 2))):
    filename = os.path.join(save_dir, 'D0096', 'wavelet', fname)
    tfr=mne.time_frequency.read_tfrs(filename)
    # Sensory-motor delay
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    tfr.plot(picks='LFPS14',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    ax.set_xticks([])
    ax.set_xlabel('')
    if fname!='Auditory-tfr.h5':
        ax.set_yticks([])
        ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'SM_Del_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)

    # Delay only
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    tfr.plot(picks='LFPS8',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'DelOnly_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)

    filename = os.path.join(save_dir, 'D0102', 'wavelet', fname)
    tfr=mne.time_frequency.read_tfrs(filename)
    # Auditory delay
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    tfr.plot(picks='RTAS5',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'Aud_Del_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)

    # Motor delay
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    tfr.plot(picks='RFPI9',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'Mtr_Del_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)
