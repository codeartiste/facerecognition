import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

data_dir = pathlib.Path("images/friends_photos/")
image_count = len(list(data_dir.glob('*.jpg')))
print(image_count)


batch_size = 32
img_height = 180
img_width = 180

seedX = 209

train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seedX,
  image_size=(img_height, img_width),
  color_mode='grayscale',
  batch_size=batch_size)
  
val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seedX,
  image_size=(img_height, img_width),
  color_mode='grayscale',
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

from tensorflow.keras import layers


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

image_batch, labels_batch = next(iter(train_ds))
image_test, labels_test = next(iter(val_ds))

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#data augumentation
'''
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
'''
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              1)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

num_classes = len(class_names)

model = tf.keras.Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
  
# simple early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience= 25, restore_best_weights=False)
  

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=4000, callbacks=[es]
)
print (model.summary())
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print('\nValidation accuracy:', test_acc)




plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
  predictions = model.predict(images[:9])
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    score = tf.nn.softmax(predictions[i])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    print(predictions[i])
    plt.title(class_names[labels[i]] + "  Predict= " +  class_names[np.argmax(predictions[i])])
    plt.axis("off")
plt.show()

model.save('friendsfacemodel') #TF model

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('friendsface_model.tflite', 'wb') as f:
  f.write(tflite_model)
