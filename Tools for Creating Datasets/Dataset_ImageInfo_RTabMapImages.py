import argparse
import os
import glob

def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)

parser = argparse.ArgumentParser(description='Image Info')
parser.add_argument("directory", help="Path to the images' directory")

args = parser.parse_args()

TOTAL_IMAGES = 700000
TOTAL_VIDEOS = 110
TOTAL_RTABMAP_IMAGES = 6363
TOTAL_DATASET_IMAGES = 2085
TOTAL_NIGHT = 1508
TOTAL_NIGHT_VIDEOS = 69
TOTAL_DAY = 577
TOTAL_DAY_VIDEOS = 41
TOTAL_AFTER_DEFOLIATION = 1283
TOTAL_BEFORE_DEFOLIATION = 802
TOTAL_90ET = 536
TOTAL_DRYLAND = 720
TOTAL_ROW21 = 689
TOTAL_ROW20 = 490
TOTAL_ROW19 = 443
TOTAL_ROW18 = 463

Day = 0
Night = 0
After = 0
Before = 0
Row_21 = 0
Row_20 = 0
Row_19 = 0
Row_18 = 0
After_Dryland = 0
After_90ET = 0

#directory = '/var/Data/Cotton Imaging Datasets/TrimmingAndCountLabel/01_26_2022/Labeled Images'

RTabMapImages = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/RTabMap Images/Images'

images = glob.glob (args.directory + "/*.jpg")

for image in images:  
    for root, subdirectories, files in os.walk(RTabMapImages):
        for subdirectory in subdirectories:
            if subdirectory == image[image.rfind('/')+1:image.rfind('_')]:

                if 'Day' in root:
                    Day += 1
                else:
                    Night += 1
                

                if subdirectory.endswith('_RTabMap_Images'):
                    After += 1
                    if len(subdirectory[subdirectory.find('_')+1:find_2nd(subdirectory, '_')]) < 3:
                        After_90ET += 1
                    else:
                        After_Dryland += 1
                else:
                    Before +=1

                if (subdirectory.startswith('Row1') or 
                    subdirectory.startswith('Row 1') or
                    subdirectory.startswith('row1') or
                    subdirectory.startswith('row 1') or
                    subdirectory.startswith('21')):
                    Row_21 += 1
                    break

                elif (subdirectory.startswith('Row2') or 
                      subdirectory.startswith('Row 2') or
                      subdirectory.startswith('row2') or
                      subdirectory.startswith('row 2') or
                      subdirectory.startswith('20')):
                    Row_20 += 1
                    break
                
                elif (subdirectory.startswith('Row3') or 
                      subdirectory.startswith('Row 3') or
                      subdirectory.startswith('row3') or
                      subdirectory.startswith('row 3') or
                      subdirectory.startswith('19')):
                    Row_19 += 1
                    break

                else:
                    Row_18 += 1
                    break

print(f"\n\n        Total Number of Videos Taken in the Field: {TOTAL_VIDEOS} --> More Than {TOTAL_IMAGES} Frames\n")
print(f"        Night = {TOTAL_NIGHT_VIDEOS} Videos\n\n        Day = {TOTAL_DAY_VIDEOS} Videos\n")
print("        ---------------------------------------------------------------------------------------------------")
print(f"        Total Number of Images used by RTabMap for 3D Reconstruction: {TOTAL_RTABMAP_IMAGES}")
print(f"        --> Which is around {TOTAL_RTABMAP_IMAGES/TOTAL_IMAGES*100:.2f}% of the Total Number of Images\n")
print("        ---------------------------------------------------------------------------------------------------")
print(f"        Total Number of Images Chosen for the Annotation Dataset: {TOTAL_DATASET_IMAGES}\n")
print(f"        --> Which is {TOTAL_DATASET_IMAGES/TOTAL_RTABMAP_IMAGES*100:.2f}% of the Total Number of RTabMap Images\n")
print(f"        Night = {TOTAL_NIGHT} Frames\n\n        Day = {TOTAL_DAY} Frames\n")
print("        ---------------------------------------------------------------------------------------------------")
print(f"        Total Number of Labeled Images = {len(images)}")
print(f"        --> Which is {len(images)/TOTAL_DATASET_IMAGES*100:.2f}% of Total Number of Images in the Annotation Dataset\n")
print(f"        Night = {Night}  --> {Night/TOTAL_NIGHT*100:.2f}% of Night Images\n\n        Day = {Day}  --> {Day/TOTAL_DAY*100:.2f}% of Day Images\n")
print("        ---------------------------------------------------------------------------------------------------")
print(f"        After Defoliation = {After}  --> {After/TOTAL_AFTER_DEFOLIATION*100:.2f}%")
print(f"            90ET = {After_90ET}  --> {After_90ET/TOTAL_90ET*100:.2f}%\n            Dryland = {After_Dryland}  --> {After_Dryland/TOTAL_DRYLAND*100:.2f}%")
print(f"        \n        Before Defoliation = {Before}  --> {Before/TOTAL_BEFORE_DEFOLIATION*100:.2f}%\n")
print("        ---------------------------------------------------------------------------------------------------")
print(f"        Row 21 = {Row_21}  --> {Row_21/TOTAL_ROW21*100:.2f}%\n        Row 20 = {Row_20}  --> {Row_20/TOTAL_ROW20*100:.2f}%")
print(f"        Row 19 = {Row_19}  --> {Row_19/TOTAL_ROW19*100:.2f}%\n        Row 18 = {Row_18}  --> {Row_18/TOTAL_ROW18*100:.2f}%\n")
print("        ---------------------------------------------------------------------------------------------------\n\n")