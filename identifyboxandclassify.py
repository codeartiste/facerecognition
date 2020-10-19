from cv2 import imread
from cv2 import imshow
from cv2 import imwrite
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import sys

# check opencv version
import cv2
# print version number
print(cv2.__version__)

directory_name = ""

print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")
    if(i == 1):
        directory_name = sys.argv[i]


import glob
files = glob.glob(directory_name + "/*.jpg")
#print(files)

class_names= ['J', 'N', 'S']


model = tf.keras.models.load_model('friendsfacemodel')
print(model.summary())

for fname in files:
    count = 0
    # load the photograph
    pixels = imread(fname)
    #percent by which the image is resized
    scale_percent = 25
    if pixels.shape[1] > 1500 :
        #calculate the 50 percent of original dimensions
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


    gray = cv2.cvtColor(scaled_pixels, cv2.COLOR_BGR2GRAY)
    # load the pre-trained model
    classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # perform face detection
    bboxes = classifier.detectMultiScale(gray, 1.25, 6)
    # print bounding box for each detected face
    for box in bboxes:
        print (box)
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        #roi = scaled_pixels[y:y2, x:x2]
        roi = gray[y:y2, x:x2]
        #Do the identification here
        print(tf.shape(roi).numpy())
        img = Image.fromarray(roi, 'L')
        img = img.resize((180, 180))
        image_sequence = img.getdata()
        image_array = np.array(image_sequence)
        #print(image_array)
        predictions = model.predict(image_array.reshape(1, 180,180,1))
        score = tf.nn.softmax(predictions[0])
        print(score)
        print(class_names[np.argmax(predictions[0])])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
        scaled_pixels,class_names[np.argmax(predictions[0])] + ' ' +
        str(int(100 * np.max(score))) + '%' ,(x2,y2), font, 1,(0,255,0),1,cv2.LINE_AA
        )
        rectangle(scaled_pixels, (x, y), (x2, y2), (0,0,255), 1)
        
            
    # show the image
    imshow('face detection', scaled_pixels)
    # keep the window open until we press a key
    waitKey(0)
# close the window
destroyAllWindows()
