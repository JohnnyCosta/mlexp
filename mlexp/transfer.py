import argparse
import os
import zipfile

import keras.backend as K
import matplotlib.pyplot as plt
from keras.applications import InceptionV3
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.lib.io import file_io
import keras

BATCH_SIZE = 20

INPUT_HEIGHT = 299
INPUT_WIDTH = 299
INPUT_MEAN = 127.5
INPUT_STD = 127.5

EPOCHS = 2

data_set = 'dataset/id'
NUM_CLASSES = 1

K.set_image_data_format('channels_last')


def model():
    original_model = InceptionV3()
    original_model.summary()

    # ______________________________________________________________________________
    # Extract needed info from pre-trained model.
    bottleneck_input = original_model.get_layer(index=0).input
    bottleneck_output = original_model.get_layer(index=-2).output
    bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

    # ______________________________________________________________________________
    # Freeze these layers so we are not retraining the full model.
    for layer in bottleneck_model.layers:
        layer.trainable = False

    # ______________________________________________________________________________
    # Build new transfer learning model.
    new_model = Sequential()
    new_model.add(bottleneck_model)

    new_model.add(Dense(NUM_CLASSES,
                        input_dim=2048,  # BOTTLENECK_DIM
                        activation='softmax'))  # Convert outputs to probabilities.
    return new_model


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
    # with tf.device('/gpu:0'):
    os.makedirs(job_dir + 'logs/', exist_ok=True)
    logs_path = job_dir + 'logs/tensorboard'

    # unzip(job_dir)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        featurewise_std_normalization=True)

    training_set = train_datagen.flow_from_directory(data_set + '/training_set',
                                                     target_size=(INPUT_WIDTH, INPUT_HEIGHT),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='binary')

    m = model()
    m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    m.summary()

    for e in range(EPOCHS):
        print('Epoch ', e)
        x_batch, y_batch = training_set.next()
        # for idx in range(len(x_batch)):
        #     plt.imshow(x_batch[idx].reshape(INPUT_WIDTH, INPUT_HEIGHT, 3))
        #     plt.show()
        labels = keras.utils.to_categorical(y_batch, num_classes=NUM_CLASSES)
        m.fit(x_batch, labels)

    #
    # # tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    #
    # history = m.fit_generator(
    #     training_set,
    #     steps_per_epoch=int(np.ceil(training_set.samples / float(batch_size))*100),
    #     epochs=epochs
    #     # validation_data=test_set,
    #     # validation_steps=int(np.ceil(test_set.samples / float(batch_size))),
    #     # workers=4,
    #     # callbacks=[tensorboard]
    # )
    #
    m.save('model.h5')
    # os.makedirs(job_dir + 'model/', exist_ok=True)
    # with file_io.FileIO('model.h5', mode='br') as input_f:
    #     with file_io.FileIO(job_dir + 'model/model.h5', mode='bw+') as output_f:
    #         output_f.write(input_f.read())
    #
    m.save_weights('weights.h5')
    # os.makedirs(job_dir + 'weights/', exist_ok=True)
    # with file_io.FileIO('weights.h5', mode='br') as input_f:
    #     with file_io.FileIO(job_dir + 'weights/weights.h5', mode='bw+') as output_f:
    #         output_f.write(input_f.read())


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
