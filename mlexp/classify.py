import argparse

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow        as tf

img_width, img_height = 299, 299
from keras.applications import InceptionV3

K.set_image_data_format('channels_last')


def read_file(file_name):
    """
    Convert string of .jpg file path to normalized np array for image processing.
    """
    file_reader = tf.read_file(file_name, "file_reader")
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)

    INPUT_HEIGHT = img_height
    INPUT_WIDTH = img_width
    INPUT_MEAN = 128
    INPUT_STD = 128

    resized = tf.image.resize_bilinear(dims_expander, [INPUT_HEIGHT, INPUT_WIDTH])
    normalized = tf.divide(tf.subtract(resized, [INPUT_MEAN]), [INPUT_STD])

    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_image(img_path, show=True):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    return img_tensor


def main(job_dir, **args):
    # model = InceptionV3()
    model = load_model(job_dir + 'model/model.h5')
    model.load_weights(job_dir + 'weights/weights.h5')


    model.summary()

    # img_path = 'dataset/predict_set/dog.jpg'
    # img_path = 'dataset/predict_set/cat.jpg'
    # img_path = 'dataset/predict_set/id.jpg'

    # new_image = load_image(img_path, show=False)

    new_image = read_file(img_path)
    # plt.imshow(new_image.reshape(img_width,img_height,3))
    # plt.show()

    pred = model.predict(new_image)

    print(pred)
    print(pred.max())
    print(pred.argmax(-1))


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
