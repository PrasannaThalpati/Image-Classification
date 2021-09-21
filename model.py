# Importing the Libraries

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# PART - 1
# DATA Preprocessing
# Preprocessing the training Set

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'D:/Its me/Deep Learning Udemy Course/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Preprocessing the Set Data

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'D:/Its me/Deep Learning Udemy Course/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# PART - 2
# Initilising the CNN

cnn = tf.keras.models.Sequential()

# Step 1 - Convolution

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[64, 64, 3]))

# Step 2 - Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second Convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening

cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output layer

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# PART - 3
# Training the CNN
# Compiling the CNN

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluting it on the Test set

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# PART - 4
# Making the Prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:/Its me/Deep Learning Udemy Course/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_1.jpg', target_size =(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)
training_set.class_indices()
if result[0][0] > 0.5:
    prediction = 'Dog'
else:
    prediction = 'Cat'

print(prediction)