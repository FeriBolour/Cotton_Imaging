import argparse
import os
import glob
import re
from shutil import copy


# For sorting the images
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]  

parser = argparse.ArgumentParser(description='Create Dataset')
parser.add_argument("directory", help="Path to the Experiments directory")
parser.add_argument("fps", type=int, help="FPS value")
#parser.add_argument("filename", help="Name of the output images")

args = parser.parse_args()

#files = glob.glob(args.directory + "/Frames/*.jpg")
files = glob.glob(args.directory + "/*.jpg")
files.sort(key=natural_sort_key)
counter = 1
#remainder = 0
Dataset_Path = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset Before Defoliation'

filename = args.directory[args.directory.rfind('/')+1:]
# print(filename)
for myFile in files:
    image_name = myFile[myFile.rfind('/') + 1:]
#    print(image_name)

    if args.fps == 15:

        if counter >= 1400 and (counter % 250 == 0):
            copy(myFile, Dataset_Path)
            os.rename(Dataset_Path + '/' + image_name,
                      Dataset_Path + '/' + filename + '_' + image_name)

    elif args.fps == 30:

        if counter >= 960 and (counter % 240 == 0):
            copy(myFile, Dataset_Path)
            os.rename(Dataset_Path + '/' + image_name,
                      Dataset_Path + '/' + filename + '_' + image_name)

    counter += 1
