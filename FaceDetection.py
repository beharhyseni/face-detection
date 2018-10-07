from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


# IMAGES' PATHS CONSTANTS
STUDENTS_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/students.jpg'


pyramid_list = []

def MakePyramid(image, minsize):
    minimum_size = 0
    pyramid_list = []
    
    with Image.open(image) as im:
        x, y = im.size
        minimum_size = min(int(x),int(y))
        
        while minimum_size>minsize:
            x = int(x * 0.75)
            y = int(y * 0.75)
            pyramid_list.append(im.resize((x,y), Image.BICUBIC))
            minimum_size = min(x,y)
    return pyramid_list
        

pyramid = MakePyramid(STUDENTS_IMAGE, 4)

for im in pyramid:
    im.show()
