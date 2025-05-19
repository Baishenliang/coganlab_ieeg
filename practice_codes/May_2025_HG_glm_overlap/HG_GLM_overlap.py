### Compare baseline ([-0.5 0]s aligned to Cue onset)
#%% Load data
import os
import pickle
# Load R^2 electrodes
import projects.GLM.glm_utils as glm
subjs, _, _, chs, times = glm.fifread('Auditory_inRep', 'zscore', 'Repeat', 'ALL')
data_LexDelay_Aud, epoc_LexDelay_Aud, _ = glm.load_stats('Auditory_inRep', 'zscore', 'Repeat', 'org_mask',
                                                         'Lexical', subjs, chs, times, 'ALL','r2')
_, _, _, _, times = glm.fifread('Resp_inRep', 'zscore', 'Repeat', 'ALL')
data_LexDelay_Resp, epoc_LexDelay_Resp, _ = glm.load_stats('Resp' + Delayseleted, 'zscore', 'Repeat', 'cluster_mask',
                                                           datasource.split('_')[1], subjs, chs, times, 'ALL')
ch_labels_roi, ch_labels = chs2atlas(subjs, data_LexDelay_Aud.labels[0])

with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data','Lex_twin_idxes_glm_BSL_correct.npy'), "rb") as f:
    Lex_twin_idxes_glm = pickle.load(f)
    with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM', 'data',
                           'Lex_twin_idxes_glm_BSL_correct.npy'), "rb") as f:
        Lex_twin_idxes_glm = pickle.load(f)
with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data','Lex_twin_idxes_hg.npy'), "rb") as f:
    Lex_twin_idxes_hg = pickle.load(f)
with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data', 'sig_idx.npy'), "rb") as f:
    Lex_glm_idxes = pickle.load(f)

#%% Plot Venn plot # Venn diagrams

from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import numpy as np

# All electrodes
glm_all = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Acoustic/all'] | Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Phonemic/all'] | Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Lexical/all']
glm_bsl_all = Lex_twin_idxes_glm['LexDelay_Aud_NoMotor_sig_idx'] | Lex_twin_idxes_glm['LexDelay_Sensorimotor_sig_idx'] | Lex_twin_idxes_glm['LexDelay_Motor_sig_idx'] | Lex_twin_idxes_glm['LexDelay_Motorprep_Only_sig_idx']
hg_bsl_all = Lex_twin_idxes_hg['LexDelay_Aud_NoMotor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Sensorimotor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Motor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Motorprep_Only_sig_idx']

plt.figure(figsize=(6, 6))
venn3([glm_all, glm_bsl_all, hg_bsl_all], ('GLM_features_corrected', 'GLM_baseline_uncorrected', 'Tstat_baseline'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap.tif'), dpi=300)
plt.close()

# Auditory/SM and Acoustic electrodes
glm_acoustic = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Acoustic/all']
glm_bsl_aud_sm = Lex_twin_idxes_glm['LexDelay_Aud_NoMotor_sig_idx'] | Lex_twin_idxes_glm['LexDelay_Sensorimotor_sig_idx']
hg_bsl_aud_sm = Lex_twin_idxes_hg['LexDelay_Aud_NoMotor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Sensorimotor_sig_idx']

plt.figure(figsize=(6, 6))
venn3([glm_acoustic, glm_bsl_aud_sm, hg_bsl_aud_sm], ('GLM_Acoustic', 'GLM_bsl_Aud_SM', 'HG_bsl_Aud_SM'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap_Aco_AudSM.tif'), dpi=300)
plt.close()

# Auditory/SM and Phonemic electrodes
glm_phonemic = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Phonemic/all']
plt.figure(figsize=(6, 6))
venn3([glm_phonemic, glm_bsl_aud_sm, hg_bsl_aud_sm], ('GLM_Phonemic', 'GLM_bsl_Aud_SM', 'HG_bsl_Aud_SM'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap_Pho_AudSM.tif'), dpi=300)
plt.close()

# Auditory/SM and Lexical Status electrodes
glm_lex = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Lexical/all']
plt.figure(figsize=(6, 6))
venn3([glm_lex, glm_bsl_aud_sm, hg_bsl_aud_sm], ('GLM_LexStatus', 'GLM_bsl_Aud_SM', 'HG_bsl_Aud_SM'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap_Lex_AudSM.tif'), dpi=300)
plt.close()

