#%%
import numpy as np 
import os 
import sys 
from glob import glob 
sys.path.append('/home/jhubadmin/Projects/autopet2023')
from metrics.metrics import (
    get_3darray_from_niftipath,
)
from organ_dictionary import organ_dict
import matplotlib.pyplot as plt 
import pandas as pd 
import time 
# %%
def calculate_intersection(lesion_mask, organ_mask, organ_index=1):
    binary_organ_mask = np.where(organ_mask == organ_index, 1, 0)
    intersection = np.sum(binary_organ_mask[lesion_mask == 1])
    return intersection

def get_useful_organs_indices(organ_dict, useful_organs):
    useful_organs_indices = [int(key) for key, value in organ_dict.items() if value in useful_organs]
    return useful_organs_indices
#%%
def get_selected_organ_mask(organarray, useful_organs):
    useful_organs_indices = get_useful_organs_indices(organ_dict, useful_organs)
    binary_mask = np.isin(organarray, useful_organs_indices)
    modified_mask_array = np.where(binary_mask, 1, 0)
    return modified_mask_array

def get_organ_removed_pet(ptarray, organarray_useful):
    ptarray_copy = ptarray.copy()
    ptarray_copy[organarray_useful == 1] = 0
    return ptarray_copy

#%%
useful_organs1 = [
    'kidney_right', 
    'kidney_left',
    'brain',
    'urinary_bladder' 
]

useful_organs2 = [
    'kidney_right', 
    'kidney_left', 
    'heart_myocardium', 
    'heart_atrium_left',
    'heart_ventricle_left',
    'heart_atrium_right',
    'heart_ventricle_right', 
    'brain', 
    'urinary_bladder'
]

useful_organs3 = [
    'spleen',
    'kidney_right', 
    'kidney_left', 
    'gallbladder',
    'liver',
    'stomach',
    'aorta',
    'pancreas',
    'heart_myocardium', 
    'heart_atrium_left',
    'heart_ventricle_left',
    'heart_atrium_right',
    'heart_ventricle_right', 
    'brain', 
    'small_bowel',
    'duodenum',
    'colon',
    'urinary_bladder'
]

useful_organs = useful_organs3 
useful_organs_indices = get_useful_organs_indices(organ_dict, useful_organs)

preddir = '/data/blobfuse/default/autopet2023_results/valid_predictions/fold0/unet/unet_fold0_randcrop192'
datadir = '/home/jhubadmin/Projects/autopet2023/create_data_split'
lym_fpath = os.path.join(datadir, 'lymphoma_metadata_fold.csv')
lun_fpath = os.path.join(datadir, 'lung_metadata_fold.csv')
mel_fpath = os.path.join(datadir, 'melanoma_metadata_fold.csv')
neg_fpath = os.path.join(datadir, 'negative_metadata_fold.csv')

lym_df = pd.read_csv(lym_fpath)
lun_df = pd.read_csv(lun_fpath)
mel_df = pd.read_csv(mel_fpath)
neg_df = pd.read_csv(neg_fpath)
fold = 0
lym_df_valid = lym_df[lym_df['FOLD'] == fold]
lun_df_valid = lun_df[lun_df['FOLD'] == fold]
mel_df_valid = mel_df[mel_df['FOLD'] == fold]
neg_df_valid = neg_df[neg_df['FOLD'] == fold]
df_valid = pd.concat([lym_df_valid, lun_df_valid, mel_df_valid, neg_df_valid], axis=0)
ids = list(df_valid['StudyID'])
#%%
organ_dir = '/data/blobfuse/default/autopet2022_data/ct_organ_segmentations/labels'
predpaths = [os.path.join(preddir, f"{id}.nii.gz") for id in ids]
organpaths = [os.path.join(organ_dir, f"{id}_0000" ,f"{id}_0000_trans.nii.gz") for id in ids]
#%%
IMAGEIDS = [os.path.basename(path)[:-7] for path in predpaths]
INTERSECTIONS = []
start = time.time()
for i in range(len(predpaths)):
    start_current = time.time()
    predarray = get_3darray_from_niftipath(predpaths[i])
    organarray = get_3darray_from_niftipath(organpaths[i])
    intersection_organs = []
    for idx in useful_organs_indices:
        inters = calculate_intersection(predarray, organarray, idx)
        intersection_organs.append(inters)
    INTERSECTIONS.append(intersection_organs)
    
    print(f"{i}:{IMAGEIDS[i]}: Intersection={intersection_organs}")
    print(f"Time taken: {(time.time() - start_current)} seconds")
print(f'Total time: {(time.time() - start)/60} mins')
#%%
data = np.column_stack((IMAGEIDS, np.vstack(INTERSECTIONS)))
organ_labels = [f'{o}_pred' for o in useful_organs]
df = pd.DataFrame(data, columns=['ImageID'] + organ_labels)
df.to_csv('organ_unetpredlesion_organ_segreggated_intersection_3.csv', index=False)
# %%
