import argparse
import os
import glob
import re
from shutil import copy


# For sorting the images
def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    int_part = int_part.split('.')[0]
    return int(int_part)
#----------------------------------------------------------
# For loading the images
def copy_images(path, directory, Dataset_Path, is_depth = False):

    if is_depth:
        files = glob.glob (path + "/*.png")
    else:
        files = glob.glob (path + "/*.jpg")
    
    files = sorted(files, key=get_key)
    counter = 1
    filename = directory[directory.rfind('/')+1:]

    for myFile in files:
        image_name = myFile[myFile.rfind('/') + 1:]
        if counter % 3 == 0:
            copy(myFile, Dataset_Path)
            os.rename(Dataset_Path + '/' + image_name,
                      Dataset_Path + '/' + filename + '_' + image_name)

        counter += 1

#----------------------------------------------------------    

rgb_dataset = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/RGB Dataset'
depth_dataset = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/Depth Dataset'

parser = argparse.ArgumentParser(description='Create Dataset')
parser.add_argument("directory", help="Path to the Experiments directory")

args = parser.parse_args()

for root, subdirectories, files in os.walk(args.directory):
    for subdirectory in subdirectories:
        if subdirectory.endswith('rgb') or subdirectory.endswith('left'):
            rgb_path = os.path.join(root, subdirectory)
            depth_path = rgb_path[:rgb_path.rfind('_')+1] + "depth"
            #rgb_path = "'" + rgb_path + "'"
            copy_images(rgb_path, rgb_path[:rgb_path.rfind('/')], rgb_dataset)
            copy_images(depth_path, rgb_path[:rgb_path.rfind('/')], depth_dataset, True)
            

