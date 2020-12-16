import numpy as np
import os
import argparse
import cv2
import skimage.io

if __name__ == '__main__':
    
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two folders one containing masks and other actual pics   
    ''')
    parser.add_argument('first_folder', help='actual pictures folder name')
    parser.add_argument('second_folder', help='masks folder name')
    # parser.add_argument('fileExt', help = 'File extension', default= '.png')
    # parser.add_argument('new_folder', help='masks folder name')
    
    args = parser.parse_args()
    # check if files in actual are more or in masks are more
    first_folder = args.first_folder
    second_folder = args.second_folder
    fileExt = '.png'
    rgb = [_ for _ in os.listdir(first_folder) if _.endswith(fileExt)] #os.listdir(first_folder)
    masks = [_ for _ in os.listdir(second_folder) if _.endswith(fileExt)] #os.listdir(second_folder)

    image = cv2.imread(os.path.join(first_folder,rgb[0]))   
    # cv2.imshow("image", image)
    # cv2.waitKey(10)
    # cv2.destroyAllWindows()

    masks.sort()
    rgb.sort()

    path = first_folder[:-8]
    if(not os.path.isdir(os.path.join(path,"semantic"))):
        os.mkdir(os.path.join(path,"semantic"))
    path = path+"/semantic"

    j = 0
    for i in range(len(rgb)):
        if (j< len(masks) and rgb[i][0:6] == masks[j][4:10]): 
            image = cv2.imread(os.path.join(second_folder,masks[j]),cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(path,rgb[i]), image)
            j += 1
        else:
            blank_image = np.zeros(image.shape, dtype=np.uint8)
            cv2.imwrite(os.path.join(path,rgb[i]),blank_image)

    

        
