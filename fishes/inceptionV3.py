#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 09:04:28 2017

@author: nehaboob
We are using keras example of using InceptionV3 to train new set of classes
we will use pretrained weights on inception data set
"""

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import backend as K
from keras import regularizers



# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
# Not working -- overfitting the data
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 8 classes
#predictions = Dense(8, activation='softmax')(x)

# add a simple network
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(256, activation='relu', W_regularizer=regularizers.l2(0.01), b_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', W_regularizer=regularizers.l2(0.01), b_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)
# and a logistic layer -- let's say we have 8 classes
predictions = Dense(8, activation='softmax')(x)


# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

# train the model on the new data for a few epochs

# get the data using generator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range=0.1,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


val_datagen = ImageDataGenerator(rescale = 1./255)

val_set = val_datagen.flow_from_directory('val_split',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            )

model.fit_generator(training_set,
                     samples_per_epoch = 3777,
                     nb_epoch = 50,
                     validation_data = None,
                     nb_val_samples = 0)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(training_set,
                     samples_per_epoch = 3019,
                     nb_epoch = 20,
                     validation_data = val_set,
                     nb_val_samples = 758)



# generating the predictions

# create the submission file
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('test_stg1',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = None,
                                            shuffle = False
                                            )
y_pred = model.predict_generator(test_set, val_samples = 1000)

import numpy as np

predicted_files = [s.replace('test/', '') for s in test_set.filenames]
final_pred = np.column_stack([predicted_files,y_pred])
np.savetxt("fishes_inceptionV3_4.csv", final_pred, 
           delimiter=",", 
           header = 'image,'+','.join(training_set.class_indices),
           comments= "", 
           fmt="%s")


from keras.models import load_model

# creates a HDF5 file 'my_model.h5'
model.save('fishes_inceptionV3_4.h5')  
# deletes the existing model
del model  

# returns a compiled model, identical to the previous one
classifier = load_model('fishes_inceptionV3_1.h5')

