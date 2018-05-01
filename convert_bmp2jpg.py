from PIL import Image
import os

inp_d = r'/home/mseals1/Documents/CNN Sensitivity/COCO DATA/Enhanced/'
d = []

for dirs, subdirs, files in os.walk(inp_d):
    for f in files:
        d.append(os.path.join(dirs, f))

for im in d:
    img = Image.open(im)

    img.save(im[:-4] + ".jpg", 'jpeg')
