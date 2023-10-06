#%%
import SimpleITK as sitk 
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.bundle import ConfigParser, download
from glob import glob  
from monai.data import decollate_batch
import time 
from monai.inferers import sliding_window_inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#%%

def read_image_array(path):
    image = sitk.ReadImage(path)
    array = np.transpose(sitk.GetArrayFromImage(image), (2,1,0))
    return array

model_name = "wholeBody_ct_segmentation"
download(name=model_name, bundle_dir=CURRENT_DIR)
# %%
model_path = os.path.join(CURRENT_DIR, 'wholeBody_ct_segmentation', 'models', 'model_lowres.pt')
config_path = os.path.join(CURRENT_DIR, 'wholeBody_ct_segmentation', 'configs', 'inference.json')
config = ConfigParser()
config.read_config(config_path)
#%%

config['device'] = 'cpu'
config['inferer']['device'] = 'cpu'
config['displayable_configs']['highres'] = False
config['displayable_configs']['sw_batch_size'] = 2
config['bundle_root'] = ''
config['dataset_dir'] = '/data/blobfuse/default/autopet2022_data/images'
ctdir = '/data/blobfuse/default/autopet2022_data/images'
ctpaths = sorted(glob(os.path.join(ctdir, '*0000.nii.gz')))
config['datalist'] = ctpaths[0:2]
output_dir = '/home/jhubadmin/Projects/ct-organs-segmentation/savedir'
config['postprocessing']['transforms'][-1]['output_dir'] = output_dir
#%%
model = config.get_parsed_content("network")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
#%%
inferer = config.get_parsed_content("inferer")
postprocessing = config.get_parsed_content("postprocessing")
dataloader = config.get_parsed_content('dataloader')
# %%

#%%
model.eval()
with torch.no_grad():
    start = time.time()
    for data in dataloader:
        start_current = time.time()
        inputs = data['image'].to(config.get_parsed_content('device'))
        roi_size = [96, 96, 96]
        sw_batch_size = 2
        data['pred'] = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
        data = [postprocessing(i) for i in decollate_batch(data)]
        elapsed_current = time.time() - start_current
        print(f"Current time: {elapsed_current/60} min")
    elapsed_total = time.time() - start 
    print(f"Total time: {elapsed_total/60} min")
# %%
output_listdir = os.listdir(output_dir)
predpaths = []
for dir in output_listdir:
    dirpath = os.path.join(output_dir, dir)
    filename = os.listdir(dirpath)[0]
    filepath = os.path.join(dirpath, filename)
    predpaths.append(filepath)
    
# %%
# plotting saved images
for i in range(len(predpaths)):
    ctpath = config['datalist'][i]
    predpath = predpaths[i]
    
    ctarray = read_image_array(ctpath)
    predarray = read_image_array(predpath)
    
    ct_cor = np.rot90(np.max(ctarray, axis=1))
    pred_cor = np.rot90(np.max(predarray, axis=1))
    
    ct_sag = np.rot90(np.max(ctarray, axis=0))
    pred_sag = np.rot90(np.max(predarray, axis=0))
    
    fig, ax = plt.subplots(2,2, figsize=(15,15))
    ax[0][0].imshow(ct_cor)
    ax[0][1].imshow(pred_cor, cmap='nipy_spectral')
    ax[1][0].imshow(ct_sag)
    ax[1][1].imshow(pred_sag, cmap='nipy_spectral')
    ax[0][0].set_title('CT', fontsize=30)
    ax[0][1].set_title('Organ Masks', fontsize=30)
    ax[0][0].set_ylabel('Coronal', fontsize=30)
    ax[1][0].set_ylabel('Sagittal', fontsize=30)
    plt.show()
    plt.close('all')
    
    
# %%
