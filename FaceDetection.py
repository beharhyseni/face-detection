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
    
    # Initialize the minimum size variable
    minimum_size = 0
    
    # Initialize the pyramid_list (to be used for storing scaled images' arrays)
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
        

# Create the pyramid of the given the given image and minimum size (min_size has to be larger than the TEMPLATE_WIDTH)
pyramid = MakePyramid(JUDY_BATS_IMAGE, 20)

        
# Calculates the width of the total pyramid image (the sum of the width of every image in the pyramid)
def pyramid_width(pyramid):
    
    # Initialize the width variable
    width = 0
    
    # Adds the width of each image in the pyramid to the variable: width, in order to obtain the total width of the pyramid
    for i in range(0, len(pyramid)):
        width += pyramid[i].size[0]
        
    # Return the width value
    return width
            
# Calculates the height of the pyramid
def pyramid_height(pyramid):
    
    # Pyramid's height is equal to the first image (original) inside of it since this image has the biggest height out of all scaled images.
    height = pyramid[0].size[1]
    
    # Return the height value 
    return height


def ShowPyramid(pyramid):
    
    # Set the width value to be equal to the width of the given pyramid
    width = pyramid_width(pyramid)
    
    # Set the height value to be equal to the height of the given pyramid
    height = pyramid_height(pyramid)
    
    # Create a new white background grayscale image with width and height equal to the whole pyramid image width and height        
    image = Image.new("L", (width, height), "white")
    
    # Initialize the temporary width and height to be used for correctly position (paste) each pyramid image into one single image
    temp_width = 0
    temp_height = 0
    
    for i in range(0, len(pyramid)):
               
        image.paste(pyramid[i],(temp_width, temp_height))
        temp_width += pyramid[i].size[0]
        
    return image


def FindTemplate(pyramid, template, threshold):
    
    # Set the TEMPLATE_WIDTH constant (=15 as given in specifications)
    TEMPLATE_WIDTH = 15
    
    # Set the output image to be the first image of the pyramid (which has the size as the original image without any scaling)
    image = pyramid[0]
    
    # Convert the image from grayscale to color in order to see the face detection rectangle in colors (="red" rectangle as required)
    image = image.convert('RGB') 
    
    # initialize the draw object on the given image
    draw = ImageDraw.Draw(image)
    
    # Open the image, here the template, so that can be accessed and used while performing the Normalized Cross-Correlation
    with Image.open(template) as template:
        
        # Set x to be the temlate's width and y to be the template's height
        x, y = template.size
        
        # Set the template width to TEMPLATE_WIDTH (= 15, as asked)
        x = TEMPLATE_WIDTH
        
        # Set the template height to a value that keeps the original ratio between width and height
        y = (float(TEMPLATE_WIDTH)/template.width) * template.height
        
        # Resize Template to width = TEMPLATE_WIDTH (= 15, as asked), and keep the ratio for the height
        template = template.resize((int(x),int(y)), Image.BICUBIC) 
        
        # Scan through every photo in the pyramid using the template
        for photo_index in range(0, len(pyramid)):       
            
            # Perform the Normalized Cross-Correcation on the given image using the given template
            output = ncc.normxcorr2D(pyramid[photo_index], template)
 
            # Save the Matrix where the normalized correlation result is above the threshold
            comparison = np.where(output>threshold)
            
            # Save the row of the Matrix where the normalized correlation result is above the threshold
            row = comparison[1]
            
            # Save the column of the Matrix where the normalized correlation result is above the threshold
            column = comparison[0]
            
            # Loop through every element in the row (or column, since they are equal in length)            
            for i,j in enumerate(row):       
                        
                # Rescale the width of the red rectangle after each iteration using the given ratio (template.width * 1/0.75^i)    
                rescaled_x = x * 1/(0.75**photo_index)
                
                # Rescale the height of the red rectangle after each iteration using the given ratio (template.height * 1/0.75^i)   
                rescaled_y = y * 1/(0.75**photo_index)  
                                       
                # Save the row element of the current iteration in M_x (name symbolizes Matrix x axis)                                        
                M_x = j
                
                # Save the column element of the current iteration in M_x (name symbolizes Matrix y axis) 
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
                
                
                
                
                
                # # SHOW THE CENTER OF X and Y
                # x1 = M_x          
                # y1 = M_y        
                # x2 = M_x        
                # y2 = M_y        
                # draw.line((x1,y1,x2,y2),fill="blue",width=10)
                # 
                # # Right line
                # x1 = M_x                
                # y1 = M_y       
                # x2 = M_x        
                # y2 = M_y        
                # draw.line((x1,y1,x2,y2),fill="blue",width=10)
                # 
                # # Top line
                # x1 = M_x        
                # y1 = M_y        
                # x2 = M_x        
                # y2 = M_y        
                # draw.line((x1,y1,x2,y2),fill="blue",width=10)
                # 
                # # Left Line
                # x1 = M_x        
                # y1 = M_y        
                # x2 = M_x        
                # y2 = M_y        
                # draw.line((x1,y1,x2,y2),fill="blue",width=10)
                            
    # Return the final image which contains red rectangles on the detected faces    
    return image
    
   
    
# Perform the FindTemplate function in the given pyramid and template image with the given threshold
FindTemplate(pyramid, TEMPLATE_IMAGE, 0.6).show()
    
    
    


