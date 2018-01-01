import csv
import os
import cv2
import tensorflow as tf
import numpy as np
import sklearn
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential


EPOCHS = 5
lines = []
data_file = "./data/driving_log.csv"
img_path = "./data/IMG/"
aug_factor = 2

#Load steering wheel data and images and create training data
samples = []
with open(data_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = img_path + batch_sample[0].split('/')[-1]
                src = cv2.imread(name)
                center_image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                center_image_flipped = np.fliplr(center_image)
                center_angle = float(batch_sample[3])
                center_angle_flipped = - center_angle
                images.append(center_image)
                images.append(center_image_flipped)
                angles.append(center_angle)
                angles.append(center_angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

"""
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split("/")[-1]
    current_path = img_path + filename
    image = cv2.imread(current_path)
    image_flipped = np.fliplr(image)
    images.append(image)
    images.append(image_flipped)
    measurement = float(line[3])
    measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
"""

# Create architecture and train model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer = "adam", loss = "mse")
model.fit_generator(train_generator, samples_per_epoch=aug_factor*len(train_samples), validation_data=validation_generator,nb_val_samples=aug_factor*len(validation_samples), nb_epoch=EPOCHS)
model.save("model.h5")
    
