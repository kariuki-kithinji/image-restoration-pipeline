import os
import torch
import torch.nn.functional as F
import torchvision as tv

import numpy as np

from .detection_models import networks
from .detection_util.util import *

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import gc


import cv2
import os

def invert_colors(file_path):
    image = cv2.imread(file_path)
    if image is not None:
        inverted_image = cv2.bitwise_not(image)
        cv2.imwrite(file_path, inverted_image)
        print(f"Inverted colors for {file_path}")
        return file_path
    else:
        print(f"Failed to read {file_path}")
        return file_path

def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "full_size":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)

def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")

def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")

class ScratchModel:
    def __init__(self,path):        
        self.model = networks.UNet(in_channels=1,out_channels=1,depth=4,conv_num=2,wf=6,padding=True,
                    batch_norm=True,up_mode="upsample",with_tanh=False,sync_bn=True,antialiasing=True,)
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(self.checkpoint["model_state"])

        #check for compatibility
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[+] using {self.device}')
        self.model.to(self.device)
        self.model.eval()
        self.input_size = "scale_256"

    def get_mask(self, path):
        
        if not os.path.isfile(path):
            print("Skipping non-file")
            return

        scratch_image = Image.open(path).convert("RGB")
        w, h = scratch_image.size

        transformed_image_PIL = data_transforms(scratch_image, self.input_size)
        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        _, _, ow, oh = scratch_image.shape
        scratch_image_scale = scale_tensor(scratch_image)

        ##check for compatibility
        scratch_image_scale = scratch_image_scale.to(self.device)

        with torch.no_grad():
            P = torch.sigmoid(self.model(scratch_image_scale))

        P = P.data.cpu()
        P = F.interpolate(P, [ow, oh], mode="nearest")

        mask_path = os.path.join(path + "_mask.png",)
        org_path = os.path.join(path + "_original.png")

        tv.utils.save_image((P >= 0.4).float(),
            mask_path,nrow=1,padding=0,normalize=True,
        )
        mask_path = invert_colors(mask_path)

        transformed_image_PIL.save(org_path)
        gc.collect()
        torch.cuda.empty_cache()

        #os.remove(path)

        return (mask_path,org_path)




