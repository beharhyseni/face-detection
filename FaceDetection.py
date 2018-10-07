from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


# IMAGES' PATHS CONSTANTS
STUDENTS_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/students.jpg'




def MakePyramid(image, minsize):
    minimum_size = 0
    pyramid_list = []
    
    with Image.open(image) as original_image:
        x, y = original_image.size
        minimum_size = min(int(x),int(y))
        pyramid_list.append(original_image.resize((x,y), Image.BICUBIC))
        
        while minimum_size>minsize:
            x = int(x * 0.75)
            y = int(y * 0.75)
             
            pyramid_list.append(original_image.resize((x,y), Image.BICUBIC))
            minimum_size = min(x,y)
    return pyramid_list
        

pyramid = MakePyramid(STUDENTS_IMAGE, 4)

# for im in pyramid:
#     im.show()
        
        
        
def pyramid_width(pyramid):
    
    width = 0
    
    for i in range(0, len(pyramid)):
        width += pyramid[i].size[0]
        
    return width
            
    
def pyramid_height(pyramid):
    
    height = pyramid[0].size[1]
           
    return height


def ShowPyramid(pyramid):
    
    width = pyramid_width(pyramid)
    height = pyramid_height(pyramid)
            
    image = Image.new("L", (width, height), "white")
    temp_width = 0
    temp_height = 0
    
    for i in range(0, len(pyramid)):
               
        image.paste(pyramid[i],(temp_width, temp_height))
        temp_width += pyramid[i].size[0]
    
    image.show()
    
ShowPyramid(pyramid)
    


