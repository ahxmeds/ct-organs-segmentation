#%%
import numpy as np 
import os 
import sys 
from glob import glob 
sys.path.append('/home/jhubadmin/Projects/autopet2023')
from metrics.metrics import (
    get_3darray_from_niftipath,
    calculate_patient_level_intersection
)
from organ_dictionary import organ_dict
import matplotlib.pyplot as plt 
import pandas as pd 
import time 
# %%
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
    'kidney_right', 
    'kidney_left', 
    'heart_myocardium', 
    'heart_atrium_left',
    'heart_ventricle_left',
    'heart_atrium_right',
    'heart_ventricle_right', 
    'brain', 
    'urinary_bladder',
    'liver',
    'spleen',
    'stomach',
    'small_bowel',
    'duodenum',
    'colon'
]

useful_organs = useful_organs3 

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
    # ptarray = get_3darray_from_niftipath(ptpaths[i])
    predarray = get_3darray_from_niftipath(predpaths[i])
    organarray = get_3darray_from_niftipath(organpaths[i])
    # organarray_copy = organarray.copy()
    organarray_useful = get_selected_organ_mask(organarray, useful_organs)
    # ptarray_copy = ptarray.copy()
    # ptarray_organ_removed = get_organ_removed_pet(ptarray, organarray_useful)
    # print(np.array_equal(ptarray, ptarray_organ_removed))
    intersection = calculate_patient_level_intersection(predarray, organarray_useful)
    INTERSECTIONS .append(intersection)
    print(f"{i}:{IMAGEIDS[i]}: Intersection={intersection}")
   

data = np.column_stack((IMAGEIDS, INTERSECTIONS ))
df = pd.DataFrame(data, columns=['ImageID', 'Intersection'])
df.to_csv('organ_unetpredlesion_intersection_3.csv', index=False)
print(f'{(time.time() - start)/60} mins')
