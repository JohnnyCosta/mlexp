import argparse
import os
import zipfile

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.layers import Input, Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.lib.io import file_io

batch_size = 256
img_width, img_height = 150, 150
epochs = 100
data_set = 'dataset'

K.set_image_data_format('channels_last')


def model(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(filters=32, kernel_size=(3, 3))(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=32, kernel_size=(3, 3))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=64, kernel_size=(3, 3))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)
    X = Dense(64)(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


def unzip(job_dir):
    print('copying file')
    with file_io.FileIO(job_dir + data_set + '.zip', mode='br') as input_f:
        with file_io.FileIO(data_set + '.zip', mode='bw+') as output_f:
            output_f.write(input_f.read())
    print('unzipping file')
    with zipfile.ZipFile(data_set + '.zip', "r") as zip_ref:
        zip_ref.extractall()
    print('finished to unzip')


def main(job_dir, **args):
    with tf.device('/gpu:0'):
        os.makedirs(job_dir + 'logs/', exist_ok=True)
        logs_path = job_dir + 'logs/tensorboard'

        unzip(job_dir)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(data_set + '/training_set',
                                                         target_size=(img_width, img_height),
                                                         batch_size=batch_size,
                                                         class_mode='binary')

        test_set = test_datagen.flow_from_directory(data_set + '/test_set',
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

        m = model((img_width, img_height, 3))

        m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        m.summary()

        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        history = m.fit_generator(
            training_set,
            steps_per_epoch=int(np.ceil(training_set.samples / float(batch_size))),
            epochs=epochs,
            validation_data=test_set,
            validation_steps=int(np.ceil(test_set.samples / float(batch_size))),
            workers=4,
            callbacks=[tensorboard]
        )

        m.save('model.h5')
        os.makedirs(job_dir + 'model/', exist_ok=True)
        with file_io.FileIO('model.h5', mode='br') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='bw+') as output_f:
                output_f.write(input_f.read())

        m.save_weights('weights.h5')
        os.makedirs(job_dir + 'weights/', exist_ok=True)
        with file_io.FileIO('weights.h5', mode='br') as input_f:
            with file_io.FileIO(job_dir + 'weights/weights.h5', mode='bw+') as output_f:
                output_f.write(input_f.read())

        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        #
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
