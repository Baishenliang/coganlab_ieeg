### Compare baseline ([-0.5 0]s aligned to Cue onset)
#%% preparation
import os
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg')
current_dir = os.getcwd()
import utils.group as gp

#%% Load data
import pickle
import json
with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data', 'sig_idx_cluster_mask.npy'), "rb") as f:
    Lex_glm_idxes = pickle.load(f)
with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data','Lex_twin_idxes_hg.npy'), "rb") as f:
    Lex_twin_idxes_hg = pickle.load(f)
with open('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\glm_config.json', 'r') as f:
    config = json.load(f)
# Extract parameters from config
Acoustic_col = config['Acoustic_col']
Phonemic_col = config['Phonemic_col']
Lexical_col = config['Lexical_col']

#%% Overlap among preonset electrodes

from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import numpy as np

# Get preonset significant electrodes
aco_pre = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Acoustic/preonset']
pho_pre = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Phonemic/preonset']
lex_pre = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Lexical/preonset']

aco = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Acoustic/aud'] | Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Acoustic/del']
pho = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Phonemic/aud'] | Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Phonemic/del']
lex = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Lexical/aud'] | Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Lexical/del']

plt.figure(figsize=(6, 6))
venn3([aco_pre, pho_pre, lex_pre], ('Acoustic preonset', 'Phonemic preonset', 'Lexical preonset'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','preonset_overlap.tif'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
venn3([aco, pho, lex], ('Acoustic', 'Phonemic', 'Lexical'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_overlap.tif'), dpi=300)
plt.close()

save_dir='D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM'
#%% Overlap among preonset electrodes and Auditory and Sensory-motor electrodes
aud = Lex_twin_idxes_hg['LexDelay_Aud_NoMotor_sig_idx']
sm = Lex_twin_idxes_hg['LexDelay_Sensorimotor_sig_idx']

plt.figure(figsize=(6, 6))
venn3([aco_pre, aud, sm], ('Acoustic pre-onset', 'Auditory', 'Sensory-motor'))
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'plot','Acoustic_preonset_overlap_withHGelec.tif'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
venn3([pho_pre, aud, sm], ('Phonemic pre-onset', 'Auditory', 'Sensory-motor'))
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'plot','Phonemic_preonset_overlap_withHGelec.tif'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
venn3([lex_pre, aud, sm], ('Lexical pre-onset', 'Auditory', 'Sensory-motor'))
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'plot','Lexical_preonset_overlap_withHGelec.tif'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
venn3([aco, aud, sm], ('Acoustic', 'Auditory', 'Sensory-motor'))
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'plot','Acoustic_overlap_withHGelec.tif'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
venn3([pho, aud, sm], ('Phonemic', 'Auditory', 'Sensory-motor'))
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'plot','Phonemic_overlap_withHGelec.tif'), dpi=300)
plt.close()

plt.figure(figsize=(6, 6))
venn3([lex, aud, sm], ('Lexical', 'Auditory', 'Sensory-motor'))
plt.tight_layout()
plt.savefig(os.path.join(save_dir,'plot','Lexical_overlap_withHGelec.tif'), dpi=300)
plt.close()

#%% GLM traces for the preonset electrodes
# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions

#Delayseleted=''
Delayseleted = '_inRep'
trial_labels='CORRECT'

# Parameters from the lexical delay task
mean_word_len=0.5#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.0 # a short period of time that we may assume auditory decay takes
delay_len=1 # from task script
motor_prep_win=[-0.25,-0.1] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[-0.1,0.75] # get windows for motor response (0.75s to avoid too much auditory feedback)
pre_stimonset_win=[-0.5,0]
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

data_LexDelay_Aud, _ = gp.load_stats(stat_type, 'Auditory' + Delayseleted, contrast, stats_root_delay, stats_root_delay)
epoc_LexDelay_Aud, _ = gp.load_stats('zscore', 'Auditory' + Delayseleted, 'epo', stats_root_delay, stats_root_delay,
                                  trial_labels=trial_labels)

Waveplot_wth=18 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title(f'HG traces for GLM sig electrdes', fontsize=20)
gp.plot_wave(epoc_LexDelay_Aud,aco,'Acoustic sig', Acoustic_col, '-', False)
gp.plot_wave(epoc_LexDelay_Aud,pho,'Phonemic sig', Phonemic_col, '-', False)
gp.plot_wave(epoc_LexDelay_Aud,lex,'Lexical sig', Lexical_col, '-', False)
plt.xlabel('Time from auditory onset (s)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.legend(fontsize=20)
plt.savefig(os.path.join(save_dir,'plot', f'HG traces for GLM sig electrdes.tif'), dpi=300, bbox_inches='tight', transparent=False)


plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title(f'HG traces for GLM preonset sig electrdes', fontsize=20)
gp.plot_wave(epoc_LexDelay_Aud,aco_pre,'Acoustic preonset sig', Acoustic_col, '-', False)
gp.plot_wave(epoc_LexDelay_Aud,aco_pre - (aud | sm),f'Acoustic preonset sig (HG not sig) N={len(aco_pre - (aud | sm))}', Acoustic_col, '--', False)
gp.plot_wave(epoc_LexDelay_Aud,aco_pre & (aud | sm),f'Acoustic preonset sig (HG sig) N={len(aco_pre & (aud | sm))}', Acoustic_col, '-.', False)
gp.plot_wave(epoc_LexDelay_Aud,pho_pre,'Phonemic preonset sig', Phonemic_col, '-', False)
gp.plot_wave(epoc_LexDelay_Aud,pho_pre- (aud | sm),f'Phonemic preonset sig (HG not sig) N={len(pho_pre - (pho | sm))}', Phonemic_col, '--', False)
gp.plot_wave(epoc_LexDelay_Aud,pho_pre & (aud | sm),f'Phonemic preonset sig (HG sig) N={len(pho_pre & (pho | sm))}', Phonemic_col, '-.', False)
gp.plot_wave(epoc_LexDelay_Aud,lex_pre,'Lexical preonset sig', Lexical_col, '-', False)
gp.plot_wave(epoc_LexDelay_Aud,lex_pre- (aud | sm),f'Lexical preonset sig (HG not sig)  N={len(lex_pre - (pho | sm))}', Lexical_col, '--', False)
gp.plot_wave(epoc_LexDelay_Aud,lex_pre & (aud | sm),f'Lexical preonset sig (HG sig)  N={len(lex_pre & (pho | sm))}', Lexical_col, '-.', False)
plt.xlabel('Time from auditory onset (s)')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.legend(fontsize=20)
plt.savefig(os.path.join(save_dir,'plot', f'HG traces for preonset sig electrdes.tif'), dpi=300, bbox_inches='tight', transparent=False)




