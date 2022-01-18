# ----- Imports -----
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import math

# ----- Base Settings -----
image_dim = 256 # width/height of images
input_shape = (image_dim, image_dim, 3) #(width, height, RGB)

epochs_per_round = 16 #The model will save after this number of epochs
rounds = 4 #The model will train for a total of epochs_per_round * rounds

batch_size = 32
learning_rate = .001

train_dir = '' #Path to directory with labeled training images
validate_dir = '' #Path to directory with labeled validation images (this is not used in the study)

save_dir = '' #Where will the models save?

# ----- Gather Settings -----
name = input("Name the model: ").replace(" ", "_")

augmented = input("Will the model train on an augmented dataset? y/n ")
augmented = (augmented == 'y')

# ----- Define training and validation datasets -----
if augmented:
    train = ImageDataGenerator(rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=.2,
        horizontal_flip=True,
    ) # Augmentations for training dataset
else:
    train = ImageDataGenerator()
    
validate = ImageDataGenerator() # Augmentations for validation dataset
train_dataset = train.flow_from_directory(train_dir, target_size=(image_dim, image_dim), class_mode='categorical') 
validate_dataset = validate.flow_from_directory(validate_dir, target_size=(image_dim, image_dim), class_mode='categorical')

n_train = train_dataset.samples # Number of images in training dataset
n_validate = validate_dataset.samples # Number of images in validation dataset

# ----- CNN Definition and Compilation -----
class_keys = list(train_dataset.class_indices.keys())
n_classes = len(class_keys)

model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(n_classes, activation='softmax')
]) # Defines the CNN layers

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
              loss='MSE',
              metrics=['accuracy'])

# ----- Save the untrained model -----
model.save(save_dir + '/' + name + '/' + name + '_0')
print(save_dir + '/' + name + '/' + name + '_0 Created!')

# ----- Train and save the model -----
for i in range(rounds):
    print('Training until ' + str((i+1) * epochs_per_round) + ' epochs')
    model.fit(train_dataset,
                        steps_per_epoch=math.floor(n_train/batch_size),
                        validation_data=validate_dataset,
                        validation_steps=math.floor(n_validate/batch_size),
                        epochs=epochs_per_round,)
    model.save(save_dir + '/' + name + '_' + str((i+1)*epochs_per_round))
    print(save_dir + '/' + name + '/' + name + '_' + str((i+1)*epochs_per_round) + ' Created!')