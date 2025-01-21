import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os


data_dir = "C:/Python/kiwi/datasets"
train_dir = "C:/Python/kiwi/datasets/train"
valid_dir = "C:/Python/kiwi/datasets/valid"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123, 
    image_size=(75, 75),
    batch_size=64
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    seed=123,
    image_size=(75, 75),
    batch_size=64
)


normalization_layer = layers.Rescaling(1. / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))


train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


model = Sequential()


model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', activation='relu', input_shape=(75, 75, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', activation='relu', input_shape=(75, 75, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    steps_per_epoch=30,
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


test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(75, 75),
    batch_size=64
)


test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print("Test Accuracy:", test_acc)
model.save('my_model.h5')

model = tf.keras.models.load_model('my_model.h5')