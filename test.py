from upscale import FaceUpsampler ,BGUpsampler


from defect import ScratchModel
from pyinpaint import Inpaint
import matplotlib.pyplot as plt

model = ScratchModel('detection/FT_Epoch_latest.pt')
mask , orginal = model.get_mask('new.webp')
inpaint = Inpaint(orginal, mask)
inpainted_img = inpaint()#*255
plt.imsave('final.png', inpainted_img)
f = FaceUpsampler()
b = BGUpsampler()