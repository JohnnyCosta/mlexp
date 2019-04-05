import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":
    model = load_model('model.h5')
    model.load_weights('weights.h5')

    # img_path = 'dataset/predict_set/June_odd-eyed-cat_cropped.jpg'
    img_path = 'dataset/predict_set/puppy-dog.jpg'
    # img_path = 'dataset/predict_set/BMW-TA.jpg'

    new_image = load_image(img_path, show=False)

    pred = model.predict(new_image)

    print('Is this a cat? %s' % pred[0])
    # print('Is this a Dog? %s' % pred[1])
