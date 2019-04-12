import argparse

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

img_width, img_height = 150, 150

K.set_image_data_format('channels_last')


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


def main(job_dir, **args):
    model = load_model(job_dir + 'model/model.h5')
    model.load_weights(job_dir + 'weights/weights.h5')

    # img_path = 'dataset/predict_set/dog.jpg'
    # img_path = 'dataset/predict_set/cat.jpg'
    img_path = 'dataset/predict_set/car.jpg'

    new_image = load_image(img_path, show=False)

    pred = model.predict(new_image)


    print('Is this a cat or dog? %s' % pred[0])


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
