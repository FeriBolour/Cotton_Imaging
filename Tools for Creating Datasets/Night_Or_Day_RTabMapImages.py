#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 00:16:57 2022

@author: farshad
"""

import argparse
import os
import glob

#parser = argparse.ArgumentParser(description='How many Night Or Day Images')
#parser.add_argument("directory", help="Path to the images' directory")

#args = parser.parse_args()

Day = 0
Night = 0

directory = '/var/Data/Cotton Imaging Datasets/TrimmingAndCountLabel/01_26_2022/Labeled Images'

RTabMapImages = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/RTabMap Images/Images'

images = glob.glob (directory + "/*.jpg")

for image in images:  
    for root, subdirectories, files in os.walk(RTabMapImages):
        for subdirectory in subdirectories:
            if subdirectory == image[image.rfind('/')+1:image.rfind('_')]:
                if 'Day' in root:
                    Day += 1
                    break
                else:
                    Night += 1
                    break

# =============================================================================
# for image in images:    
#     for root, dirs, files in os.walk(RTabMapImages):
#         for file in files:
#             if file == image[image.rfind('_')+1:]:
#                 if 'Day' in root:
#                     Day += 1
#                     break
#                 else:
#                     Night += 1
#                     break
# =============================================================================
                
print(f"Night = {Night}\n\nDay = {Day}")