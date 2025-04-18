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
import json
import pickle

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
chs=data_LexDelay_Aud.labels[0]
# Load coordinates
chs_coor=gp.get_coor(chs)
# Get Auditory, Sensory-motor, and motor electrode index
with open(os.path.join('data', 'LexDelay_twin_idxes.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)
chs_coor['Auditory']=chs_coor.index.isin(LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx']).astype(int)
chs_coor['Sensory-motor']=chs_coor.index.isin(LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx']).astype(int)
chs_coor['Motor']=chs_coor.index.isin(LexDelay_twin_idxes['LexDelay_Motor_sig_idx']).astype(int)
# clean electrodes
chs_coor_c = chs_coor[
    (chs_coor[['Auditory', 'Sensory-motor', 'Motor']].sum(axis=1)) > 0].reset_index(drop=True)
# recode
chs_coor_c = label_electrode_type(chs_coor_c)

#%% get neighbour profile:
#%% chaos
df = chs_coor_c

# Step 1: Get the subframe from each subject
subjects = df['subj'].unique()
subject_frames = {subj: df[df['subj'] == subj] for subj in subjects}


# Step 2: Calculate the distance between each electrode
def calculate_distances(df):
    coords = df[['x', 'y', 'z']].values
    dist_matrix = cdist(coords, coords)
    return dist_matrix


# Step 3: Find neighbouring electrodes within a threshold
def find_neighbours(dist_matrix, threshold=5):
    neighbours = []
    for i, row in enumerate(dist_matrix):
        neighbours_i = np.where(row < threshold)[0]  # Indices of electrodes with distance lower than threshold
        neighbours.append(neighbours_i)
    return neighbours


# Step 4: Count the number of electrodes in each type
def count_types(df):
    type_counts = df['type'].value_counts()
    return type_counts


# Step 5: Get the neighbouring electrode types
def get_neighbouring_types(neighbours, df):
    neighbour_types = []
    for i, neighbour_indices in enumerate(neighbours):
        types = df.iloc[neighbour_indices]['type'].values
        neighbour_types.append(types)
    return neighbour_types


# Main processing for each subject
threshold = 5
subject_results = {}

for subj, sub_df in subject_frames.items():
    # Step 2: Calculate the distance matrix
    dist_matrix = calculate_distances(sub_df)

    # Step 3: Find neighbouring electrodes
    neighbours = find_neighbours(dist_matrix, threshold)

    # Step 4: Count electrode types
    type_counts = count_types(sub_df)

    # Step 5: Get neighbouring electrode types
    neighbouring_types = get_neighbouring_types(neighbours, sub_df)

    # Store the results for each subject
    subject_results[subj] = {
        'type_counts': type_counts,
        'neighbouring_types': neighbouring_types
    }

# Now subject_results holds the information for each subject
subject_results