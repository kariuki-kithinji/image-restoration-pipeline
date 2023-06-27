import sys
import logging
import os
import urllib.request

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)

from deoldify._device import _Device

device = _Device()

import os

def download_model():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    file_path = './models/ColorizeStable_gen.pth'
    
    if os.path.exists(file_path):
        print('Model file already exists.')
        return
    
    command = f"wget -O {file_path} https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=0"
    
    try:
        os.system(command)
        print('File downloaded successfully.')
    except Exception as e:
        print('An error occurred while downloading the file:', str(e))

    # Downloading and extracting another model
    checkpoint_file = 'global_checkpoints.zip'
    command = f"wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/{checkpoint_file}"
    
    try:
        os.system(command)
        print('Checkpoint file downloaded successfully.')
        
        # Extracting the downloaded ZIP file
        os.system(f"unzip {checkpoint_file}")
        print('Checkpoint file extracted successfully.')
    except Exception as e:
        print('An error occurred while downloading or extracting the checkpoint file:', str(e))