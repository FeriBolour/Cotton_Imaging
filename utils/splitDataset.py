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
parser.add_argument("directory", help="Path to the Dataset directory")

args = parser.parse_args()

#files = glob.glob(args.directory + "/Frames/*.jpg")
files = glob.glob(args.directory + "/*.jpg")
files.sort(key=natural_sort_key)
counter = 0

Ethan = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/Splitted Dataset/Ethan'
Farshad = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/Splitted Dataset/Farshad'
Irish = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/Splitted Dataset/Irish'
Long = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/Splitted Dataset/Long'
Yildirim = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/Dataset For Annotation/Splitted Dataset/Yildirim'

Paths = [Ethan, Farshad, Irish, Long, Yildirim]

for myFile in files:
    copy(myFile, Paths[counter%5])
    counter += 1