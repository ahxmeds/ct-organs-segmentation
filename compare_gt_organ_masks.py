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
    # ptarray = get_3darray_from_niftipath(ptpaths[i])
    gtarray = get_3darray_from_niftipath(gtpaths[i])
    organarray = get_3darray_from_niftipath(organpaths[i])
    # organarray_copy = organarray.copy()
    organarray_useful = get_selected_organ_mask(organarray, useful_organs)
    # ptarray_copy = ptarray.copy()
    # ptarray_organ_removed = get_organ_removed_pet(ptarray, organarray_useful)
    # print(np.array_equal(ptarray, ptarray_organ_removed))
    intersection = calculate_patient_level_intersection(gtarray, organarray_useful)
    INTERSECTIONS.append(intersection)
    print(f"{i}:{IMAGEIDS[i]}: Intersection={round(intersection, 4)}")
    # if np.array_equal(ptarray, ptarray_organ_removed):
    #     print('Nothing removed')
    # else:
    #     print("Something removed")
    # print('\n')
    
    # fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.7)
    # pt_cor = np.rot90(np.max(ptarray, axis=1))
    # gt_cor = np.rot90(np.max(gtarray, axis=1))
    # organuseful_cor = np.rot90(np.max(organarray_useful, axis=1))
    # ptorganremoved_cor = np.rot90(np.max(ptarray_organ_removed, axis=1))
    # ax[0].imshow(pt_cor, cmap='Spectral_r')
    # ax[1].imshow(gt_cor, cmap='Spectral_r')
    # ax[2].imshow(organuseful_cor, cmap='Spectral_r')
    # ax[3].imshow(ptorganremoved_cor, cmap='Spectral_r')
    
    # ax[0].set_title('Original PET')
    # ax[1].set_title('Lesion GT')
    # ax[2].set_title('Organ Labels')
    # ax[3].set_title('PET with organs removed')
    # plt.show()
    # plt.close('all')

data = np.column_stack((IMAGEIDS, INTERSECTIONS))
df = pd.DataFrame(data, columns=['ImageID', 'Intersection'])
df.to_csv('organ_gtlesion_intersection_3.csv', index=False)
print(f'{(time.time() - start)/60} mins')
# %%
# ax[0].imshow(np.rot90(ptarray[:, j, :]), cmap='Spectral_r')
# ax[1].imshow(np.rot90(gtarray[:, j, :]), cmap='Spectral_r')
# ax[2].imshow(np.rot90(organarray_useful[:, j, :]), cmap='Spectral_r')
# ax[3].imshow(np.rot90(ptarray_organ_removed[:, j, :]), cmap='Spectral_r')