#!/usr/bin/env python3

import argparse
import os
import glob

parser = argparse.ArgumentParser(description='How many Before Or After Defoliation Images')
parser.add_argument("directory", help="Path to the images' directory")

args = parser.parse_args()

#directory = '/var/Data/Cotton Imaging Datasets/TrimmingAndCountLabel/01_26_2022/Labeled Images'

RTabMapImages = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/RTabMap Images/Images'

After = 0
Before = 0

images = glob.glob (args.directory + "/*.jpg")

for image in images:  
    for root, subdirectories, files in os.walk(RTabMapImages):
        for subdirectory in subdirectories:
            if subdirectory == image[image.rfind('/')+1:image.rfind('_')]:
                if subdirectory.endswith('_RTabMap_Images'):
                    After += 1
                    break
                else:
                    Before +=1
                    break


print(f"\nAfter Defoliation = {After}\n\nBefore Defoliation = {Before}\n")