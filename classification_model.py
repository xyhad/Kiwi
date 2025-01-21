import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import BatchNormalization

import matplotlib.pyplot as plt
import os


data_dir = "C:/Python/kiwi/datasets"
train_dir = "C:/Python/kiwi/datasets/train"
valid_dir = "C:/Python/kiwi/datasets/valid"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,          # Normalize images to [0, 1]
    rotation_range=20,         # Randomly rotate images
    width_shift_range=0.2,     # Randomly shift images horizontally
    height_shift_range=0.2,    # Randomly shift images vertically
    shear_range=0.2,           # Randomly shear images
    zoom_range=0.2,            # Randomly zoom into images
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # How to fill in newly created pixels
)

val_datagen = ImageDataGenerator(rescale=1. / 255)  # Only rescale for validation

# New code using ImageDataGenerator
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(75, 75),
    batch_size=64,
    class_mode='sparse',  # for sparse categorical labels
    seed=123
)

val_dataset = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(75, 75),
    batch_size=64,
    class_mode='sparse',  # for sparse categorical labels
    seed=123
)


#normalization_layer = layers.Rescaling(1. / 255)
#train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
#val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))


learning_rate = 0.0005  # You can adjust this value as needed
optimizer = Adam(learning_rate=learning_rate)

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu', input_shape=(75, 75, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', activation='relu', input_shape=(75, 75, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))  
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',   
    patience=5,          
    restore_best_weights=True  
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=35,
    steps_per_epoch=40,
    callbacks=[early_stopping]  
)


loss, accuracy = model.evaluate(val_dataset)
print("Validation Accuracy:", accuracy)
print("Number of epochs completed:", len(history.history['accuracy']))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1]) 
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1])  # Adjust the range dynamically
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

test_dir = "C:/Python/kiwi/datasets/test"


test_dataset = val_datagen.flow_from_directory(
    test_dir,
    target_size=(75, 75),
    batch_size=25,
    class_mode='sparse',  # for sparse categorical labels
    seed=123
)


#test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
#test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print("Test Accuracy:", test_acc)
model.save('my_model.h5')

model = tf.keras.models.load_model('my_model.h5')