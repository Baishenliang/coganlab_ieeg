#%% Introduction
# This script is made for the comparison in HG traces among Repeat vs. YesNo, Delay vs. NoDelay, and Word vs. Nonword
import os
import pickle
# check if currently running a slurm job

N_cores=-1
HOME = os.path.expanduser("~")
if HOME=='C:\\Users\\bl314':
    is_cluster=False
    task_ID=-1
elif HOME=='/hpc/home/bl314':
    is_cluster = True
    task_ID= int(os.environ['SLURM_ARRAY_TASK_ID'])
else:
    raise ValueError('unrecognized home directory')
if is_cluster:
    LAB_root = os.path.join(HOME, "workspace")
    script_dir = os.path.dirname(os.path.join(LAB_root,'coganlab_ieeg\\projects\\PCA_LDA\\pca_lda.py'))
    current_dir = os.getcwd()
    if current_dir != script_dir:
        os.chdir(script_dir)
    sf_dir = os.path.join(LAB_root,'PCA_LDA_results')
    with open(os.path.join(sf_dir, f'Lex_twin_idxes_hg.npy'), "rb") as f:
        LexDelay_twin_idxes = pickle.load(f)
else:
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\PCA_LDA\\pca_lda.py')
    current_dir = os.getcwd()
    if current_dir != script_dir:
        os.chdir(script_dir)
    sf_dir = 'results'
    with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
        LexDelay_twin_idxes = pickle.load(f)

import sys
if not is_cluster:
    sys.path.append(os.path.abspath(os.path.join("..", "..")))
    import utils.group as gp
from ieeg.decoding.decode import classes_from_labels,Decoder
from ieeg.calc.fast import mixup
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from scipy.stats import norm

# %% function block
mean_word_len=0.65#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1.125 # average length from sound offset to Go onset
num_perm=500

def get_time_indexs(time_str,start_float:float=0,end_float:float=delay_len):
    start_idx = np.searchsorted(time_str, start_float, side='left')
    end_idx = np.searchsorted(time_str, end_float, side='right')
    indices = list(range(start_idx, end_idx))
    return indices

def calculate_acc(conf_matrix_rates: np.ndarray):
    conf_matrix_rates_np = np.array(conf_matrix_rates)
    diagonal_array = np.diag(conf_matrix_rates_np)
    return np.mean(diagonal_array)

def calculate_d_prime(conf_matrix_rates: np.ndarray) -> float:

    hit_rate = conf_matrix_rates[0, 0]
    false_alarm_rate = conf_matrix_rates[1, 0]

    if hit_rate >= 1:
        hit_rate = 1 - 1e-6
    if hit_rate <= 0:
        hit_rate = 1e-6

    if false_alarm_rate >= 1:
        false_alarm_rate = 1 - 1e-6
    if false_alarm_rate <= 0:
        false_alarm_rate = 1e-6

    d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)

    return d_prime
# %% groups of patients
from pickle import FALSE

datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
#groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
# For lexical delay task, whether run the data only with repeat tasks
trial_labels='CORRECT'

# %% Sort data and get significant electrode lists
import os
import numpy as np
import matplotlib.pyplot as plt
from ieeg.arrays.label import LabeledArray

if not is_cluster:
    if os.path.exists(os.path.join(sf_dir,'epoc_LexDelayRep_Aud.npy')):
        epoc_LexDelayRep_Aud = LabeledArray.fromfile(os.path.join(sf_dir,'epoc_LexDelayRep_Aud'))
    else:
        stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
        stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

        fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','LexicalDecRepDelay','group')
        if not os.path.exists(os.path.join(fig_save_dir)):
            os.mkdir(os.path.join(fig_save_dir))

        stats_save_root = os.path.join(stats_root_delay,'group')
        if not os.path.exists(os.path.join(stats_save_root)):
            os.mkdir(os.path.join(stats_save_root))

        if groupsTag=="LexDelay":
            epoc_LexDelayRep_Aud,_=gp.load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=True)
            epoc_LexDelayRep_Aud.tofile(os.path.join(sf_dir,'epoc_LexDelayRep_Aud'))


# %% Select electrodes
read_mode=True
run_decoder=False
debug_mode=True
loaded_data={}
task_i=0
for feature_tag in ('lexstus','pho1'):
    for t_tag,t_range in zip(
            ('encode','delay'),
            ([0,mean_word_len+auditory_decay],[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len])
    ):
        if read_mode:
            loaded_data[t_tag] = {}
        for elec_grp,elec_idx in zip(
                ('Motor_delay','Auditory_delay','Sensorymotor_delay','Delay_only'),
                ('LexDelay_Motor_in_Delay_sig_idx','LexDelay_Auditory_in_Delay_sig_idx','LexDelay_Sensorimotor_in_Delay_sig_idx','LexDelay_DelayOnly_sig_idx')
        ):
            task_i+=1
            if debug_mode and ((feature_tag!='pho1') or (t_tag!='encode') or (elec_grp!='Auditory_delay')):
                continue
            print(f'task_i {task_i}')
            if is_cluster and task_i!=task_ID:
                continue
            else:
                print(f'Now Doing {t_tag} {elec_grp}')
            if os.path.exists(os.path.join(sf_dir, f'epoc_LexDelayRep_Aud_{feature_tag}_{t_tag}_{elec_grp}.npy')):
                m = LabeledArray.fromfile(os.path.join(sf_dir, f'epoc_LexDelayRep_Aud_{feature_tag}_{t_tag}_{elec_grp}'))
            elif is_cluster:
                raise ValueError('Gao shen me a lian ge bian liang dou mei you!')
            else:
                m_chs=epoc_LexDelayRep_Aud.take(list(LexDelay_twin_idxes[elec_idx]),axis=1)
                m=m_chs.take(get_time_indexs(m_chs.labels[2],t_range[0],t_range[1]),axis=2)
                if feature_tag=='lexstus':
                    cats, labels = classes_from_labels(m.labels[0], '/', 2)
                elif feature_tag=='pho1':
                    cats, labels = classes_from_labels(m.labels[0], '/', 3,crop=0)
                    not_small_trials = np.where(~np.isin(labels, [2,7,16,17]))[0].tolist()
                    m=m.take(not_small_trials,axis=0)
                elif feature_tag=='pho2':
                    cats, labels = classes_from_labels(m.labels[0], '/', 3,crop=1)
                    not_small_trials = np.where(labels!=4)[0].tolist()
                    m=m.take(not_small_trials,axis=0)
                mixup(m, 0)
                m.tofile(os.path.join(sf_dir, f'epoc_LexDelayRep_Aud_{feature_tag}_{t_tag}_{elec_grp}'))
            if feature_tag=='lexstus':
                cats, labels = classes_from_labels(m.labels[0], '/', 2)
                n_split=10
            elif feature_tag=='pho1':
                cats, labels = classes_from_labels(m.labels[0], '/', 3, crop=0)
                n_split=5
            elif feature_tag=='pho2':
                cats, labels = classes_from_labels(m.labels[0], '/', 3, crop=1)
                n_split=10
            if run_decoder:
                # decoder = Decoder(cats, oversample=True, n_splits=5, n_repeats=100)
                decoder = Decoder(cats, n_splits=n_split, n_repeats=500)
                cm = decoder.cv_cm(m.__array__().swapaxes(0,1), labels, normalize='true',n_jobs=N_cores)
                cm_dprime = calculate_acc(cm)
    
                # Shuffle
                np.random.seed(42)
                cm_perm_dist = []
                cm_perm_dprime_dist = []
    
                decoder_perm = Decoder(cats, n_splits=n_split, n_repeats=1)
                for i in range(num_perm):
                    print(f'runing perm {i} in {num_perm}')
                    labels_shuffle = np.random.permutation(labels)
                    cm_perm = decoder_perm.cv_cm(m.__array__().swapaxes(0, 1), labels_shuffle, normalize='true', n_jobs=N_cores)
                    cm_perm_dist.append(cm_perm)
                    cm_perm_dprime_dist.append(calculate_acc(cm_perm))
    
                count_greater_equal = sum(1 for val in cm_perm_dprime_dist if val >= cm_dprime)
                total_count = len(cm_perm_dprime_dist)
                p_value = (count_greater_equal / total_count) * 100
    
                data_to_save = {
                    'cm': cm,
                    'cm_dprime': cm_dprime,
                    'cm_perm_dist': cm_perm_dist,
                    'cm_perm_dprime_dist': cm_perm_dprime_dist,
                    'p_value': p_value
                }
    
                filename = os.path.join(sf_dir, f'{feature_tag}_{elec_grp}_{t_tag}.pkl')
                with open(filename, 'wb') as f:
                    pickle.dump(data_to_save, f)
            if read_mode:
                filename = os.path.join(sf_dir, f'{feature_tag}_{elec_grp}_{t_tag}.pkl')
                with open(filename, 'rb') as f:
                    ld=pickle.load(f)
                    loaded_data[t_tag][elec_grp] = ld
                    org_acc=np.mean([loaded_data[t_tag][elec_grp]['cm'][0,0],loaded_data[t_tag][elec_grp]['cm'][1,1]])
                    cm_perm_dist=loaded_data[t_tag][elec_grp]['cm_perm_dist']
                    perm_accs=[np.mean([cm_perm_dist[i][0,0],cm_perm_dist[i][1,1]]) for i in range(len(cm_perm_dist))]
                    print(loaded_data[t_tag][elec_grp]['cm'])
                    print(f'acc: {org_acc}, p: {sum(org_acc<perm_accs)/len(perm_accs)}')
                    print(f'dprime: {loaded_data[t_tag][elec_grp]['cm_dprime']},p: {loaded_data[t_tag][elec_grp]['p_value']/100}')
    
                    fig, ax = plt.subplots(figsize=(7,5))
                    plt.rcParams['font.size'] = 20
                    disp = ConfusionMatrixDisplay(confusion_matrix=loaded_data[t_tag][elec_grp]['cm'], display_labels=cats.keys())
                    cm_plot=disp.plot(colorbar=False, ax=ax)
                    im = cm_plot.im_
                    im.set_clim(vmin=0.44, vmax=0.56)
                    plt.title(f'{elec_grp}_{t_tag}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(sf_dir,f'{feature_tag}_{elec_grp}_{t_tag}.tif'), dpi=300)
                    plt.close()