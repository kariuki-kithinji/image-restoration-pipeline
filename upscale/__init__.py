from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os

from basicsr.utils import imwrite
from gfpgan import GFPGANer

import cv2

class BGUpsampler:
    def __init__(self):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.upsampler = RealESRGANer(
                scale=2, 
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=self.model,tile=400,tile_pad=10,pre_pad=0,half=True)  # need to set False in CPU mode

class FaceUpsampler:
    def __init__(self):
        self.arch = 'clean'
        self.channel_multiplier = 2
        self.bg_upsampler = BGUpsampler().upsampler
        self.model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        self.restorer = GFPGANer(
                model_path=self.model_path,upscale=2,arch=self.arch,
                channel_multiplier=self.channel_multiplier,bg_upsampler=self.bg_upsampler)
    
    def upsample(self,path):
        basename, ext = os.path.splitext(path)
        input_img = cv2.imread(path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
                input_img,has_aligned=False,
                only_center_face=False,paste_back=True,weight=0.5)
        
        imwrite(restored_img, basename+'_upsampled.png')

