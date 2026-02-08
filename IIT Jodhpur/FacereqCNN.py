import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
import os

num_classes= 7
img_rows, img_cols = 48, 48
batch_size = 8

import zipfile

with zipfile.ZipFile(r'C:\\Users\\tejas\Downloads\\FRE-1 (2).zip', 'r') as zip_ref:
    zip_ref.extractall(r'C:\\Users\\tejas\Downloads\\FRE-1')

train_data_dir = train_data_dir = r'C:\\Users\\tejas\Downloads\\FRE-1\\train'


test_data_dir = r'C:\\Users\\tejas\Downloads\\FRE-1\\test'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.3, zoom_range=0.3, width_shift_range=0.4,height_shift_range=0.4, horizontal_flip=True, vertical_flip=True )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory( train_data_dir, target_size=(img_rows, img_cols), batch_size=batch_size, class_mode='categorical', color_mode='grayscale', shuffle=True)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

model= Sequential()

#Block 1
model.add(Conv2D(32,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#Block 2
model.add(Conv2D(64,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#Block 3
model.add(Conv2D(128,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#Block 4
model.add(Conv2D(256,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#Block 5
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block 6
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block 7
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoints= ModelCheckpoint('models/facereq_cnn.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,patience=3, verbose=1, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)

callbacks = [checkpoints, early_stopping, reduce_lr]
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

nb_train_samples = train_generator.samples
nb_test_samples = test_generator.samples
epochs = 25

history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, validation_data=test_generator, validation_steps=nb_test_samples // batch_size, callbacks=callbacks)



