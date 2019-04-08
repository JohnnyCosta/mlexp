import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


batch_size = 256
img_width, img_height = 150, 150
epochs = 10

if __name__ == "__main__":
    model = load_model('model.h5')
    model.load_weights('weights.h5')

    model.summary()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(img_width, img_height),
                                                     batch_size=batch_size,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode='binary')
    history = model.fit_generator(
        training_set,
        steps_per_epoch=int(np.ceil(training_set.samples / float(batch_size)))*2,
        epochs=epochs,
        validation_data=test_set,
        validation_steps=int(np.ceil(test_set.samples / float(batch_size))),
        workers=4
    )


    model.save('model.h5')
    model.save_weights('weights.h5')

    # from IPython.display import Inline
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
