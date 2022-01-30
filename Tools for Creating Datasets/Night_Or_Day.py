import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


#parser = argparse.ArgumentParser(description='Count Night and Day Images')
#parser.add_argument("directory", help="Path to direcotry for Night and Day images")

#args = parser.parse_args()

directory = '/var/Data/Cotton Imaging Datasets/Annotation Dataset/RTabMap Images/Images/18_9_bot_RTabMap_Images/18_9_bot_left'

Night = 0
Day = 0

images = glob.glob (directory + "/*.jpg")

for image in images:
    
    img = cv2.imread(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    n, b = np.histogram(hsv[:, :, 0].flatten())
    h_max = b[np.where(n == n.max())]

    n, b = np.histogram(hsv[:, :, 1].flatten())
    s_max = b[np.where(n == n.max())]

    n, b = np.histogram(hsv[:, :, 2].flatten())
    v_max = b[np.where(n == n.max())]

    h_lTop_std = np.std(hsv[:150, :150, 0].flatten())

    n, b = np.histogram(hsv[:150, :150, 1].flatten())
    s_lTop_max = b[np.where(n == n.max())]

    n, b = np.histogram(hsv[:150, :150, 2].flatten())
    v_lTop_max = b[np.where(n == n.max())]
    
    n, b = np.histogram(hsv[100:200, 1200:, 1].flatten())
    s_rTop_max = b[np.where(n == n.max())]

    n, b = np.histogram(hsv[100:200, 1200:, 2].flatten())
    v_rTop_max = b[np.where(n == n.max())]

    if h_max == 0 and s_max == 0 and v_max < 220:
        Night += 1

    elif v_lTop_max > 220:
        if s_lTop_max == 0 or h_lTop_std > 70:
            Day += 1
            print(image)
    
    elif s_rTop_max == 0 and v_rTop_max > 220:
        Day += 1
        print(image)

    else:
        Night += 1


print(f"Night = {Night}\n\nDay = {Day}")
# print(hsv.shape)          # -> Dimsensions of the Matrix
# plt.hist(hsv[:,:,0].flatten())        # -> flatten : turn matrix to a vector
# np.mean(np.mean(hsv, axis = 0), axis = 0)  # mean row-wise: mean for each chanel
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))      # -> change BGR to RGB for display purposes    