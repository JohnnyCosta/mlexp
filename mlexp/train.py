import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

batch_size = 256
img_width, img_height = 150, 150

if __name__ == "__main__":
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

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
        steps_per_epoch=int(np.ceil(training_set.samples / float(batch_size))),
        epochs=15,
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
