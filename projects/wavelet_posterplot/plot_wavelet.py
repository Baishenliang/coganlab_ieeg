import os
import mne
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
import matplotlib.pyplot as plt
import numpy as np

font_scale=1.5
plt.rcParams['font.size'] = 14*font_scale
plt.rcParams['axes.titlesize'] = 16*font_scale
plt.rcParams['axes.labelsize'] = 12*font_scale
plt.rcParams['xtick.labelsize'] = 12*font_scale
plt.rcParams['ytick.labelsize'] = 12*font_scale
plt.rcParams['legend.fontsize'] = 12*font_scale

HOME = os.path.expanduser("~")
Task_Tag="LexicalDecRepDelay"
save_dir=os.path.join(HOME, "Box", "CoganLab", "D_Data",Task_Tag,"Baishen_Figs","LexicalDecRepDelay")

for fname,size,ax_space in zip(
        ('Auditory-tfr.h5','Resp-tfr.h5'),
        ((12, 4),(12*1.5/2, 4)),
        ([0.15, 0.15, 0.7, 0.7],[0.15, 0.15, 0.7, 0.7])):
    filename = os.path.join(save_dir, 'D0096', 'wavelet', fname)
    tfr=mne.time_frequency.read_tfrs(filename)

    # Sensory-motor delay
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(ax_space)
    tfr.plot(picks='LFPS14',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    if fname=='Auditory-tfr.h5':
        ax.set_xlim([-0.25,1.5])
        ax.set_xticks(np.arange(-0.25,1.75, 0.25))
        ax.set_yticks([6,19,67,300,1065])
    else:
        ax.set_xlim([-0.25,1])
        ax.set_xticks(np.arange(-0.25, 1.05, 0.25))
    # ax.set_xticks([])
    # ax.set_xlabel('')
    if fname!='Auditory-tfr.h5':
        ax.set_yticks([])
        ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'SM_Del_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)

    # Delay only
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(ax_space)
    tfr.plot(picks='LFPS8',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    if fname=='Auditory-tfr.h5':
        ax.set_xlim([-0.25,1.5])
        ax.set_xticks(np.arange(-0.25,1.75, 0.25))
        ax.set_yticks([6,19,67,300,1065])
    else:
        ax.set_xlim([-0.25,1])
        ax.set_xticks(np.arange(-0.25, 1.05, 0.25))
    # ax.set_xlabel('')
    # ax.set_yticks([])
    # ax.set_ylabel('')
    if fname!='Auditory-tfr.h5':
        ax.set_yticks([])
        ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'DelOnly_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)

    filename = os.path.join(save_dir, 'D0102', 'wavelet', fname)
    tfr=mne.time_frequency.read_tfrs(filename)
    # Auditory delay
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(ax_space)
    tfr.plot(picks='RTAS5',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    if fname=='Auditory-tfr.h5':
        ax.set_xlim([-0.25,1.5])
        ax.set_xticks(np.arange(-0.25,1.75, 0.25))
        ax.set_yticks([6,19,67,300,1065])
    else:
        ax.set_xlim([-0.25,1])
        ax.set_xticks(np.arange(-0.25, 1.05, 0.25))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    if fname!='Auditory-tfr.h5':
        ax.set_yticks([])
        ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'Aud_Del_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)

    # Motor delay
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(ax_space)
    tfr.plot(picks='RFPI9',vlim=(-2,2),cmap=parula_map,axes=ax,colorbar=False)
    plt.axvline(x=0, linestyle='--', color='k')
    if fname=='Auditory-tfr.h5':
        ax.set_xlim([-0.25,1.5])
        ax.set_xticks(np.arange(-0.25,1.75, 0.25))
        ax.set_yticks([6,19,67,300,1065])
    else:
        ax.set_xlim([-0.25,1])
        ax.set_xticks(np.arange(-0.25, 1.05, 0.25))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    if fname!='Auditory-tfr.h5':
        ax.set_yticks([])
        ax.set_ylabel('')
    fig.savefig(os.path.join(save_dir, 'group',f'Mtr_Del_elec_exmp_{fname.split('-')[0]}.tif'), dpi=300)
