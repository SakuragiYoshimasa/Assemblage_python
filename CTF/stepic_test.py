#coding: utf-8
import stepic
from PIL import Image
img = Image.open('output/zip/00000186/share1.png')
res = stepic.decode(img)
print(res)
