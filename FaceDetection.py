from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


# IMAGES' PATHS CONSTANTS
STUDENTS_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/students.jpg'
JUDY_BATS_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/judybats.jpg'
FANS_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/fans.jpg'
TREE_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/tree.jpg'
FAMILY_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/family.jpg'
TEMPLATE_IMAGE = 'C:/Users/behar/Desktop/Face-Detection-in-a-Scaled-Representation/libs/faces/template.jpg'





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
        

pyramid = MakePyramid(JUDY_BATS_IMAGE, 20)

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
    return image


def FindTemplate(pyramid, template, threshold):
    
    TEMPLATE_WIDTH = 15
    image = pyramid[0]
    image = image.convert('RGB') 
    draw = ImageDraw.Draw(image)
    with Image.open(template) as template:
        x, y = template.size
        x = TEMPLATE_WIDTH
        y = (float(TEMPLATE_WIDTH)/template.width) * template.height
        template = template.resize((int(x),int(y)), Image.BICUBIC) 
        
        
        for photo_index in range(0, len(pyramid)):       
            
            print photo_index
            output = ncc.normxcorr2D(pyramid[photo_index], template)
 
            comparison = np.where(output>threshold)
            row = comparison[1]
            column = comparison[0]
            
            for i,j in enumerate(row):       
                        
                rescaled_x = x * 1/(0.75**photo_index)
                rescaled_y = y * 1/(0.75**photo_index)  
                                              
                M_x = j
                M_y = column[i]
                
                
                # THE FACE DETECTION RECTANGLE BOX
                # Bottom line
                x1 = M_x        - rescaled_x  
                y1 = M_y        + rescaled_y   
                x2 = M_x        + rescaled_x        
                y2 = M_y        + rescaled_y               
                draw.line((x1,y1,x2,y2),fill="red",width=2)
                
                # Right line
                x1 = M_x        + rescaled_x        
                y1 = M_y        + rescaled_y     
                x2 = M_x        + rescaled_x       
                y2 = M_y        - rescaled_y                  
                draw.line((x1,y1,x2,y2),fill="red",width=2)
                
                # Top line
                x1 = M_x        - rescaled_x
                y1 = M_y        - rescaled_y
                x2 = M_x        + rescaled_x
                y2 = M_y        - rescaled_y                  
                draw.line((x1,y1,x2,y2),fill="red",width=2)
                
                # Left Line
                x1 = M_x        - rescaled_x 
                y1 = M_y        + rescaled_y 
                x2 = M_x        - rescaled_x 
                y2 = M_y        - rescaled_y                        
                draw.line((x1,y1,x2,y2),fill="red",width=2)
                            
            
    return image
    
   
    
    
FindTemplate(pyramid, TEMPLATE_IMAGE, 0.6).show()
    
    
    


