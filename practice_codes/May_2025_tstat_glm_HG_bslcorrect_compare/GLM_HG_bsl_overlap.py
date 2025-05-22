### Compare baseline ([-0.5 0]s aligned to Cue onset)
#%% Load data
import os
import pickle
with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data','sig_idx_org_mask.npy'), "rb") as f:
    Lex_glm_uncorrected_idxes = pickle.load(f)
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
glm_uncorrected_all = Lex_glm_uncorrected_idxes['Auditory_inRep/Repeat/ALL/Acoustic/all'] | Lex_glm_uncorrected_idxes['Auditory_inRep/Repeat/ALL/Phonemic/all'] | Lex_glm_uncorrected_idxes['Auditory_inRep/Repeat/ALL/Lexical/all']
hg_bsl_all = Lex_twin_idxes_hg['LexDelay_Aud_NoMotor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Sensorimotor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Motor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Motorprep_Only_sig_idx']

plt.figure(figsize=(6, 6))
venn3([glm_all, glm_uncorrected_all, hg_bsl_all], ('GLM_fea (corrected)', 'GLM_fea (uncorrected)', 'Tstat_baseline'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap.tif'), dpi=300)
plt.close()

# Auditory/SM and Acoustic electrodes
glm_aco_corrected = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Acoustic/all']
glm_aco_uncorrected = Lex_glm_uncorrected_idxes['Auditory_inRep/Repeat/ALL/Acoustic/all']
hg_bsl_aud_sm = Lex_twin_idxes_hg['LexDelay_Aud_NoMotor_sig_idx'] | Lex_twin_idxes_hg['LexDelay_Sensorimotor_sig_idx']
hg_bsl_m = Lex_twin_idxes_hg['LexDelay_Motor_sig_idx']

plt.figure(figsize=(6, 6))
venn3([glm_aco_corrected, glm_aco_uncorrected, hg_bsl_aud_sm], ('GLM_Aco_Corrected', 'GLM_Aco_UnCorrected', 'HG_bsl_Aud_SM'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap_Aco_AudSM.tif'), dpi=300)
plt.close()

# Auditory/SM and Phonemic electrodes
glm_pho_corrected = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Phonemic/all']
glm_pho_uncorrected = Lex_glm_uncorrected_idxes['Auditory_inRep/Repeat/ALL/Phonemic/all']
plt.figure(figsize=(6, 6))
venn3([glm_pho_corrected, glm_pho_uncorrected, hg_bsl_aud_sm], ('GLM_Pho_Corrected', 'GLM_Pho_UnCorrected', 'HG_bsl_Aud_SM'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap_Pho_AudSM.tif'), dpi=300)
plt.close()

# Auditory/SM and Lexical Status electrodes
glm_lex_corrected = Lex_glm_idxes['Auditory_inRep/Repeat/ALL/Lexical/all']
glm_lex_uncorrected = Lex_glm_uncorrected_idxes['Auditory_inRep/Repeat/ALL/Lexical/all']
plt.figure(figsize=(6, 6))
venn3([glm_lex_corrected, glm_lex_uncorrected, hg_bsl_aud_sm], ('GLM_Lex_Corrected', 'GLM_Lex_UnCorrected', 'HG_bsl_Aud_SM'))
plt.tight_layout()
plt.savefig(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','plot','glm_tstat_overlap_Lex_AudSM.tif'), dpi=300)
plt.close()

