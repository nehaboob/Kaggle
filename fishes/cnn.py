# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# CNN
# input conv relu conv relu pool relu conv relu pool fully connected

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'sigmoid'))

# Compiling the CNN
# Change the loss funtion to categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_stg1',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = None,
                                            )

classifier.fit_generator(training_set,
                         steps_per_epoch = 3777,
                         epochs = 1,
                         validation_data = None,
                         nb_val_samples = None)

# generating the predictions
y_pred = classifier.predict_generator(test_set, steps = 32, verbose = 1)

# create the submission file
import numpy as np

predicted_files = [s.replace('test/', '') for s in test_set.filenames]
final_pred = np.column_stack([predicted_files,y_pred])
np.savetxt("fishes_1.csv", final_pred, 
           delimiter=",", 
           header = 'image,'+','.join(training_set.class_indices),
           comments= "", 
           fmt="%s")


from keras.models import load_model

# creates a HDF5 file 'my_model.h5'
classifier.save('fishes_cnn_1.h5')  
# deletes the existing model
del classifier  

# returns a compiled model, identical to the previous one
classifier = load_model('fishes_cnn_1.h5')

