import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    history = model.fit_generator(
        training_set,
        steps_per_epoch=500,
        epochs=3,
        validation_data=test_set,
        validation_steps=10)

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
