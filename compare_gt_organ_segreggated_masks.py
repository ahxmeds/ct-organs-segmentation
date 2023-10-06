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

dir = '/data/blobfuse/default/autopet2022_data/'
ptpaths = sorted(glob(os.path.join(dir, 'images', '*0001.nii.gz')))
gtpaths = sorted(glob(os.path.join(dir, 'labels', '*.nii.gz')))
organ_dir = '/data/blobfuse/default/autopet2022_data/ct_organ_segmentations/labels'
organ_dirlist = sorted(os.listdir(organ_dir))
organpaths = [os.path.join(organ_dir, organ_dirlist[i], f'{organ_dirlist[i]}_trans.nii.gz') for i in range(len(organ_dirlist))]

#%%
IMAGEIDS = [os.path.basename(path)[:-7] for path in gtpaths]
INTERSECTIONS = []
start = time.time()
for i in range(len(gtpaths)):
    start_current = time.time()
    gtarray = get_3darray_from_niftipath(gtpaths[i])
    organarray = get_3darray_from_niftipath(organpaths[i])
    intersection_organs = []
    for idx in useful_organs_indices:
        inters = calculate_intersection(gtarray, organarray, idx)
        intersection_organs.append(inters)
    INTERSECTIONS.append(intersection_organs)
    
    print(f"{i}:{IMAGEIDS[i]}: Intersection={intersection_organs}")
    print(f"Time taken: {(time.time() - start_current)} seconds")
print(f'Total time: {(time.time() - start)/60} mins')
  
#%%
data = np.column_stack((IMAGEIDS, np.vstack(INTERSECTIONS)))
organ_labels = [f'{o}_gt' for o in useful_organs]
df = pd.DataFrame(data, columns=['ImageID'] + organ_labels)
df.to_csv('organ_gtlesion_organ_segreggated_intersection_3.csv', index=False)
