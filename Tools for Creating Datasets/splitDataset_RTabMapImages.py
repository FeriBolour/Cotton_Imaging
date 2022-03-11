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
def copy_images(path, directory, Dataset_Path, pathsCounter, is_depth = False):

    if is_depth:
        files = glob.glob (path + "/*.png")
    else:
        files = glob.glob (path + "/*.jpg")
    
    files = sorted(files, key=get_key)
    counter = 0
    filename = directory[directory.rfind('/')+1:]

    for myFile in files:
        image_name = myFile[myFile.rfind('/') + 1:]
        if counter % 3 == 0:
            copy(myFile, Dataset_Path[pathsCounter%5])
            os.rename(Dataset_Path[pathsCounter%5] + '/' + image_name,
                      Dataset_Path[pathsCounter%5] + '/' + filename + '_' + image_name)

        counter += 1
        pathsCounter += 1
        
    return pathsCounter

#----------------------------------------------------------    

parser = argparse.ArgumentParser(description='Split Dataset')
parser.add_argument("directory", help="Path to the Experiments directory")

args = parser.parse_args()

Ethan = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/Splitted Dataset 3/Ethan/Ethan'
Farshad = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/Splitted Dataset 3/Farshad/Farshad'
Irish = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/Splitted Dataset 3/Irish/Irish'
Long = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/Splitted Dataset 3/Long/Long'
Yildirim = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/RTabMap Images Dataset/Splitted Dataset 3/Yildirim/Yildirim'

RGB_Paths = [Ethan, Farshad, Irish, Long, Yildirim]
Depth_Paths = [Ethan + '_Depth', Farshad + '_Depth', Irish + '_Depth', Long + '_Depth', Yildirim + '_Depth']

pathsCounter = 0

for root, subdirectories, files in os.walk(args.directory):
    for subdirectory in subdirectories:
        if subdirectory.endswith('rgb') or subdirectory.endswith('left'):
            rgb_path = os.path.join(root, subdirectory)
            depth_path = rgb_path[:rgb_path.rfind('_')+1] + "depth"
            copy_images(rgb_path, rgb_path[:rgb_path.rfind('/')], RGB_Paths, pathsCounter)
            pathsCounter = copy_images(depth_path, rgb_path[:rgb_path.rfind('/')], Depth_Paths, pathsCounter, True)

