from itertools import islice, chain

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from itertools import cycle

from keras import Input
from keras.applications import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D, concatenate
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.metrics import mean_absolute_error

from flow_dataframe import flow_from_dataframe

tstart = datetime.now()

# hyperparameters
NUM_EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VAL = 16


# default size of InceptionResNetV2
IMG_SIZE = (299, 299)

print('==================================================')
print('============ Preprocessing Image Data ============')
print('==================================================')

print('current time: %s' % str(datetime.now()))

# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).

core_idg = ImageDataGenerator(zoom_range=0.2,
                              fill_mode='nearest',
                              featurewise_center=False,  # set input mean to 0 over the dataset
                              samplewise_center=False,  # set each sample mean to 0
                              featurewise_std_normalization=False,  # divide inputs by std of the dataset
                              samplewise_std_normalization=False,  # divide each input by its std
                              zca_whitening=False,  # apply ZCA whitening
                              rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
                              width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                              height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                              horizontal_flip=True,  # randomly flip images
                              vertical_flip=False)

val_idg = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)

print('==================================================')
print('============ Creating Data Generators ============')
print('==================================================')

print('current time: %s' % str(datetime.now()))

print('==================================================')
print('========== Reading RSNA Boneage Dataset ==========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

class_str_col = 'boneage'
gender_str_col = 'male'

base_bone_dir = os.path.join('rsna-bone-age')
boneage_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))
boneage_df['path'] = boneage_df['id'].map(lambda x: os.path.join(base_bone_dir,
                                                         'boneage-training-dataset', 
                                                         'boneage-training-dataset', 
                                                         '{}.png'.format(x)))

boneage_df['exists'] = boneage_df['path'].map(os.path.exists)
print(boneage_df['exists'].sum(), 'images found of', boneage_df.shape[0], 'total')

boneage_df[gender_str_col] = boneage_df[gender_str_col].map(lambda x: np.array([1]) if x else np.array([0])) # map boolean values to 1 and 0

train_df_boneage, valid_df_boneage = train_test_split(boneage_df, test_size=0.2,
                                                      random_state=2018)  # ,stratify=boneage_df['boneage_category'])
print('train', train_df_boneage.shape[0], 'validation', valid_df_boneage.shape[0])

train_gen_boneage = flow_from_dataframe(core_idg, train_df_boneage, path_col='path', y_col=class_str_col,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb', batch_size=BATCH_SIZE_TRAIN)

# used a fixed dataset for evaluating the algorithm
valid_gen_boneage = flow_from_dataframe(core_idg, valid_df_boneage, path_col='path', y_col=class_str_col,
                                        target_size=IMG_SIZE,
                                        color_mode='rgb',
                                        batch_size=BATCH_SIZE_VAL)  # we can use much larger batches for evaluation

print('==================================================')
print('================= Building Model =================')
print('==================================================')

print('current time: %s' % str(datetime.now()))

i1 = Input(shape=(299, 299, 3), name='input_img')
i2 = Input(shape=(1,), name='input_gender')
base = InceptionV3(input_tensor=i1, input_shape=(299, 299, 3), include_top=False, weights=None)

feature_img = base.get_layer(name='mixed10').output
feature_img = AveragePooling2D((2, 2))(feature_img)
feature_img = Flatten()(feature_img)
feature_gender = Dense(32, activation='relu')(i2)
feature = concatenate([feature_img, feature_gender], axis=1)

# feature = feature_img
o = Dense(1000, activation='relu')(feature)
o = Dense(1000, activation='relu')(o)
o = Dense(1)(o)
model = Model(inputs=[i1, i2], outputs=o)
optimizer = Adam(lr=1e-3)
model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mae'])

print('==================================================')
print('======= Training Model on Boneage Dataset ========')
print('==================================================')

print('current time: %s' % str(datetime.now()))

model.summary()

weight_path = "{}_weights.best.hdf5".format('bone_age')

#checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
#                             save_best_only=True, mode='min', save_best_only=True,)

early = EarlyStopping(monitor="val_loss", mode="min",
                      patience=10)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, verbose=1,
                                   save_best_only=True, mode='auto', epsilon=0.0001, cooldown=5)


def combined_generators(image_generator, gender_data, batch_size):
    gender_generator = cycle(batch(gender_data, batch_size))
    while True:
        nextImage = next(image_generator)
        nextGender = next(gender_generator)
        assert len(nextImage[0]) == len(nextGender)
        yield [nextImage[0], nextGender], nextImage[1]


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


train_gen_wrapper = combined_generators(train_gen_boneage, train_df_boneage[gender_str_col], BATCH_SIZE_TRAIN)
val_gen_wrapper = combined_generators(valid_gen_boneage, valid_df_boneage[gender_str_col], BATCH_SIZE_VAL)
print('batch size: ', BATCH_SIZE_TRAIN)

history = model.fit_generator(train_gen_wrapper, validation_data=val_gen_wrapper,
                              epochs=NUM_EPOCHS, steps_per_epoch=len(train_gen_boneage),
                              validation_steps=len(valid_gen_boneage),
                              callbacks=[early, reduceLROnPlat])

model.save('saved_model.h5')

print('Model Saved ! :) \n')

print('Boneage dataset (final): val_mean_absolute_error: ', history.history['val_mean_absolute_error'][-1])

print('==================================================')
print('================ Evaluating Model ================')
print('==================================================')

tend = datetime.now()
print('current time: %s' % str(datetime.now()))
print('elapsed time: %s' % str((tend - tstart)))
