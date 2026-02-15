# Get the distribution of neighbouring electrode types of each electrode
#%% Import everything
import os

# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# set plotting themes
#sns.set_theme(style="ticks", palette="pastel")
sns.set_theme(style="ticks")
cm = 1/2.54
plt.rcParams['savefig.dpi']=300
plt.rcParams['font.size']=10
plt.rcParams['axes.linewidth']=1
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['font.family']=['Arial']
plt.rcParams['text.color']='black'
savefig_format='.pdf'
datasource='hg'


Sensorimotor_col = [1, 0, 0]  # Sensorimotor (Red)
Auditory_col = [0, 1, 0]  # Auditory (Green)
Motor_col = [0, 0, 1]  # Motor (Blue)

mode='count'#count: count the number of neighbouring electrodes, dist: calculate the averaged distance for neighbouring electrodes
def neigh_func_dist(d):
    d_in=d['distance']
    if len(d_in)==0:
        return np.nan
    else:
        return np.nanmin(d_in).astype(float)
if mode == 'count':
    neigh_func = len
elif mode == 'dist':
    neigh_func = neigh_func_dist

get_coord_method='group'
# Method to get coordinate, and do neighbouring analyses.
# - individual: project to individual space, and do analyses only in individual electrodes
# - group: project to fs_average group space, and do analyses in all individuals' electrodes
if get_coord_method=='individual':
    ind_tag='sub'
elif get_coord_method=='group':
    ind_tag='grp'
#%% functions

def label_electrode_type(df):
    def get_type(row):
        if row['Auditory'] == 1:
            return 'Auditory'
        elif row['Sensory-motor'] == 1:
            return 'Sensory-motor'
        elif row['Motor'] == 1:
            return 'Motor'
    df['type'] = df.apply(get_type, axis=1)
    df = df.drop(columns=['Auditory', 'Sensory-motor', 'Motor'])
    return df

#%% load electrodes, coordinates, and indexs
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
data_LexDelay_Aud,subjs=gp.load_stats('mask','Auditory_inRep','ave',stats_root_delay,stats_root_delay)

#Get electrodes in hickok SM ROIs
ch_labels_roi,ch_labels=gp.chs2atlas(subjs,data_LexDelay_Aud.labels[0])
hickok_roi_sets={'All':set(),'lIFG':set(),'lIPL':set(),'Spt':set(),'lPMC':set(),'Wgw_a55b':set(),'Wgw_p55b':set()}

chs=data_LexDelay_Aud.labels[0]
# Load coordinates
chs_coor=gp.get_coor(chs,get_coord_method)
hickok_roi_labels=gp.hickok_roi_sphere(chs_coor)
for i,item in enumerate(hickok_roi_labels[0].values()):
    hickok_roi_sets['All'].add(i)
    if item!='N/A':
        hickok_roi_sets[item].add(i)
# Get Auditory, Sensory-motor, and motor electrode index
with open(os.path.join('data', f'Lex_twin_idxes_{datasource}.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)

#for roi in ['All','lIFG','lIPL','Spt','lPMC']:
for roi in ['All',]:

    #% get Auditory, Sensory-motor, and Motor electrodes
    chs_coor['Auditory']=chs_coor.index.isin(LexDelay_twin_idxes['LexDelay_Auditory_in_Delay_sig_idx'] & hickok_roi_sets[roi]).astype(int)
    chs_coor['Sensory-motor']=chs_coor.index.isin(LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx'] & hickok_roi_sets[roi]).astype(int)
    chs_coor['Motor']=chs_coor.index.isin(LexDelay_twin_idxes['LexDelay_Motor_in_Delay_sig_idx'] & hickok_roi_sets[roi]).astype(int)
    # clean electrodes
    chs_coor_c = chs_coor[
        (chs_coor[['Auditory', 'Sensory-motor', 'Motor']].sum(axis=1)) > 0].reset_index(drop=True)
    # recode
    chs_coor_c = label_electrode_type(chs_coor_c)

    #% get neighbour profile:
    dist_thres=15 #mm
    def euclidean_distance(row1, row2):
        return np.sqrt((row1['x'] - row2['x']) ** 2 + (row1['y'] - row2['y']) ** 2 + (row1['z'] - row2['z']) ** 2)

    chs_coor_c['N_Auditory'] = 0
    chs_coor_c['N_SM'] = 0
    chs_coor_c['N_Motor'] = 0

    if get_coord_method=='individual':
        for subj in chs_coor_c['subj'].unique():
            subj_df = chs_coor_c[chs_coor_c['subj'] == subj]
            for index, electrode in subj_df.iterrows():
                distances = []
                for idx, other_electrode in subj_df.iterrows():
                    if index != idx:
                        distance = euclidean_distance(electrode, other_electrode)
                        distances.append((distance, other_electrode['type']))
                distance_df = pd.DataFrame(distances, columns=['distance', 'type'])
                chs_coor_c.at[index, 'Neib_Auditory'] = neigh_func(
                    distance_df[(distance_df['distance'] <= dist_thres) & (distance_df['type'] == 'Auditory')])
                chs_coor_c.at[index, 'Neib_SM'] = neigh_func(
                    distance_df[(distance_df['distance'] <= dist_thres) & (distance_df['type'] == 'Sensory-motor')])
                chs_coor_c.at[index, 'Neib_Motor'] = neigh_func(
                    distance_df[(distance_df['distance'] <= dist_thres) & (distance_df['type'] == 'Motor')])
    elif get_coord_method=='group':
        for index, electrode in chs_coor_c.iterrows():
            distances = []
            print(electrode)
            for idx, other_electrode in chs_coor_c.iterrows():
                if index != idx:
                    distance = euclidean_distance(electrode, other_electrode)
                    distances.append((distance, other_electrode['type']))
            distance_df = pd.DataFrame(distances, columns=['distance', 'type'])
            chs_coor_c.at[index, 'Neib_Auditory'] = neigh_func(
                distance_df[(distance_df['distance'] <= dist_thres) & (distance_df['type'] == 'Auditory')])
            chs_coor_c.at[index, 'Neib_SM'] = neigh_func(
                distance_df[(distance_df['distance'] <= dist_thres) & (distance_df['type'] == 'Sensory-motor')])
            chs_coor_c.at[index, 'Neib_Motor'] = neigh_func(
                distance_df[(distance_df['distance'] <= dist_thres) & (distance_df['type'] == 'Motor')])

    #% plot neighbouring
    #change data to long format
    hue_order = ['Neib_Auditory', 'Neib_SM', 'Neib_Motor']
    chs_coor_c['subj_label'] = chs_coor_c['subj'] + '-' + chs_coor_c['label']
    chs_coor_c_l = chs_coor_c.melt(id_vars=['subj_label', 'type'], value_vars=hue_order,
                                  var_name='neighbor_type', value_name='count')
    #plot
    plt.figure(figsize=(8 * cm, 8 * cm))

    boxplot_colors= [Auditory_col, Auditory_col, Auditory_col, Sensorimotor_col, Sensorimotor_col, Sensorimotor_col,Motor_col,Motor_col,Motor_col]
    stripplot_colors = [Auditory_col, Sensorimotor_col, Motor_col, Auditory_col, Sensorimotor_col, Motor_col,Auditory_col, Sensorimotor_col, Motor_col]

    if mode=='count':
        ytitles = ['No. Electrodes']
        #subtitles = [f'Neighboring Electrodes for Elec. in {roi}']
        subtitles = [f'Neighboring Electrodes for each Elec.']
    elif mode=='dist':
        ytitles = ['Distance (mm)']
        subtitles = [f'Distance of Neighboring Elec. for Elec. in {roi}']
    x_order = ['Auditory', 'Sensory-motor', 'Motor']
    F_name = 'results/Exp3_easyhard' + savefig_format

    if roi=='All' and mode=='count':
        if get_coord_method == 'individual':
            y_limits = [(-0.1, 25)]  # Specify different y-axis limits for each subplot
            y_ticks = [range(0, 26, 5)]
        elif get_coord_method == 'group':
            y_limits = [(-0.1, 60)]
            y_ticks = [range(0, 60, 10)]
    elif mode=='count':
        if get_coord_method == 'individual':
            y_limits = [(-0.1, 6)]
            y_ticks = [range(0, 7, 1)]
        elif get_coord_method == 'group':
            y_limits = [(-0.1, 25)]
            y_ticks = [range(0, 26, 2)]
    elif mode=='dist':
        y_limits = [(-0.1, dist_thres-1)]
        y_ticks = [range(0, dist_thres, 1)]
    for i, var in enumerate(['count'], start=1):
        plt.subplot(1, 1, i)

        if roi=='All':
            fillbar=True
        else:
            fillbar=False

        barbar = sns.barplot(x='type', y=var, errorbar=None, data=chs_coor_c_l, hue='neighbor_type', hue_order=hue_order,order=x_order, saturation=1,
                             fill=fillbar, alpha=1, linewidth=0.8, capsize=0.1, zorder=1)

        if roi=='All':
            j = 0
            for patch in barbar.patches:
                patch.set_facecolor(boxplot_colors[j])
                j = j + 1
                if j == 9:
                    break

        # ax=sns.boxplot(x='Group', y=var, data=data,showfliers=False, hue='Group',order=x_order,saturation=1)
        sns.despine()

        if mode=='count':

            stripstrip = sns.stripplot(chs_coor_c_l, x="type", y=var, size=1, hue='neighbor_type', hue_order=hue_order, alpha=1, jitter=0.3, linewidth=0.1,
                                       edgecolor='white', order=x_order, zorder=2, dodge=True)
            # stripstrip.legend(title='Neighbor Type', loc='upper right')

            if roi == 'All':
                for k in range(9):
                    path_collection = stripstrip.collections[k]
                    path_collection.set_facecolor(stripplot_colors[k])

            gp.bsliang_add_connecting_lines(plt, 0, stripstrip)
            gp.bsliang_add_connecting_lines(plt, 1, stripstrip)
            gp.bsliang_add_connecting_lines(plt, 3, stripstrip)
            gp.bsliang_add_connecting_lines(plt, 4, stripstrip)
            gp.bsliang_add_connecting_lines(plt, 6, stripstrip)
            gp.bsliang_add_connecting_lines(plt, 7, stripstrip)

        # Choice 3: bar plot with fill - errbar
        ax2 = sns.barplot(x='type', y=var, errorbar='se', data=chs_coor_c_l, hue='neighbor_type', hue_order=hue_order,order=x_order, saturation=1,
                          fill=False, alpha=0.5, linewidth=0, capsize=0.1, err_kws={'linewidth': 0.8, 'color': 'black'},
                          zorder=3)

        plt.xlabel('')
        plt.ylabel(ytitles[i - 1], y=gp.bsliang_align_yaxis(y_limits[i - 1], y_ticks[i - 1]))
        plt.ylim(y_limits[i - 1])
        plt.yticks(y_ticks[i - 1])
        plt.title(subtitles[i - 1], y=1.2)
        plt.gca().tick_params(axis='x', direction='in', length=0, labelrotation=45)
        plt.gca().tick_params(axis='y', direction='in', length=2)
        plt.gca().get_legend().remove()

    plt.tight_layout(w_pad=1.5)
    plt.savefig(os.path.join('..','Neighbour_electrode','plot',f'Neighbour_ele_{roi}_{mode}_{ind_tag}.tif'), dpi=300,bbox_inches='tight', transparent=False)

    #%% stats
    from scipy.stats import ttest_ind
    for nei_type in hue_order:
        Aud=chs_coor_c_l[(chs_coor_c_l['type'] == 'Auditory') & (chs_coor_c_l['neighbor_type'] == nei_type)]['count']
        SM=chs_coor_c_l[(chs_coor_c_l['type'] == 'Sensory-motor') & (chs_coor_c_l['neighbor_type'] == nei_type)]['count']
        Mot=chs_coor_c_l[(chs_coor_c_l['type'] == 'Motor') & (chs_coor_c_l['neighbor_type'] == nei_type)]['count']
        t_stat, p_value = ttest_ind(Aud, SM ,nan_policy='omit')
        print(f"Aud vs. SM in {nei_type} for {roi}: {t_stat}, p-value: {p_value}")
        t_stat, p_value = ttest_ind(SM, Mot, nan_policy='omit')
        print(f"SM vs. Mot in {nei_type} for {roi}: {t_stat}, p-value: {p_value}")
        t_stat, p_value = ttest_ind(Aud, Mot, nan_policy='omit')
        print(f"Aud vs. Mot in {nei_type} for {roi}: {t_stat}, p-value: {p_value}")
        print("=========================================================================")
