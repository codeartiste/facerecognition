# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import imwrite
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

import os
import sys
import cv2
import glob

# check opencv version
# print version number
print(cv2.__version__)

directory_name = "" # If you don't give an argument populate this

print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")
    if(i == 1):
        directory_name = sys.argv[i]



files = glob.glob(directory_name + "/*.jpg")
#Debug print
print(files)

for fname in files:
    count = 0
    # load the photograph
    pixels = imread(fname)
    #percent by which the image is resized
    scale_percent = 25
    if pixels.shape[1] > 1500 :
        #calculate the scaled percent of original dimensions
        width = int(pixels.shape[1] * scale_percent / 100)
        height = int(pixels.shape[0] * scale_percent / 100)
        print(width)
        print(height)
        # dsize
        dsize = (width, height)
        # resize image
        scaled_pixels = cv2.resize(pixels, dsize)
    else:
        scaled_pixels = pixels


    # Note detection works on gray images
    gray = cv2.cvtColor(scaled_pixels, cv2.COLOR_BGR2GRAY)
    # load the pre-trained model
    classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # perform face detection
    bboxes = classifier.detectMultiScale(gray, 1.22, 6)
    # print bounding box for each detected face
    for box in bboxes:
        print (box)
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # cut a rectangle over the pixels
        roi = scaled_pixels[y:y2, x:x2]
        imwrite('images/rects/'+ os.path.split(fname)[-1] +'_' +str(count) +'.jpg', roi)
        #rectangle(scaled_pixels, (x, y), (x2, y2), (0,0,255), 2)
        count = count + 1
     
    #Debug a
    #show the image
    #imshow('face detection', scaled_pixels)
    # keep the window open until we press a key
    #waitKey(0)
# close the window
destroyAllWindows()
