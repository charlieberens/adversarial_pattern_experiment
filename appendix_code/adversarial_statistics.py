# ----- Imports ----- 
import tensorflow as tf
import random
from os import walk
import pandas as pd

# ----- Config Settings ----- 
model_dir = '' # Where are the models saved
models = ['augmented', 'not_augmented'] # What are the models named
epochs = [0,16,32,48,64] # What epochs will be tested

image_dir = '' # Where are the validation images saved
data_output_dir = '' # Where will the data export

epsilons = [0, .1, .2, .3, .4] #What epsilons will be used. 0 
sample_size = 16 # Samples per class
image_dim = 256 # Dimensions of the images

# ----- Dataset Info -----
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
validate_folder_size = 200

# ----- Generates which images will be used. Represented as a 2d array, the first dimension referring to class and the second referring to the index of the image within the class directory-----
index_matrix = []

for animal in classes:
    indices = random.sample(range(validate_folder_size), sample_size)
    index_matrix.append(indices)

# ----- Given an animal and image-index, this function will return the raw image file -----
def get_image(animal, index):
    directory = image_dir + '/' + animal
    filenames = next(walk(directory), (None, None, []))[2] #Black magic that returns a list of filenames in directory
    return tf.io.read_file(directory + '/' + filenames[index])

# ----- Using FGSM, this function generates the adversarial pattern -----
def get_perterbations(model, input_image, class_index):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = tf.keras.losses.MSE(class_index, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

# ------ Given a list of confidence values and index of the correct class, this calculates MSE ------
def get_loss(probs, animal_index): #Calculates MSE loss
    total = 0
    for index, prob in enumerate(probs):
        if index == animal_index:
            total += (1-prob)*(1-prob)
        else:
            total += prob*prob
    return total/len(probs)

# ------- This function evaluates a model on the evalation images modified with all epsilon values (including eps = 0). It returns the average loss and accuracy for each epsilon, and the list of every loss for every image and espilon (including eps=0)
def evaluate_model(model):    
    eps_all_losses = [[] for __ in epsilons] # [[]] * len(epsilons) makes a list of lists that reference eachother. Editing one affects all of them.
    eps_losses = [0] * len(epsilons)
    eps_accuracy = [0] * len(epsilons)

    for animal_index, indices in enumerate(index_matrix):
        animal_name = classes[animal_index]
        
        for index in indices:
            # Load and process image
            image_raw = get_image(animal_name, index)
            image = tf.image.decode_image(image_raw, channels=3)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, (256,256))
            image = image[None, ...] #adds that goofy empty dimension

            perterbations = get_perterbations(model, image, animal_index)

            # Check new predicted value for each epsilon
            for eps_index, eps in enumerate(epsilons):
                image_probs = model.predict(image - perterbations * eps)[0]

                loss = get_loss(image_probs, animal_index)
                eps_losses[eps_index] += loss
                eps_all_losses[eps_index].append(loss)

                # If correct guess -> add one to total
                if image_probs[animal_index] == max(image_probs):
                    eps_accuracy[eps_index] += 1

    eps_avg_losses = [total / (len(classes) * sample_size) for total in eps_losses]
    eps_avg_accuracy = [total / (len(classes) * sample_size) for total in eps_accuracy]

    return (eps_avg_losses, eps_avg_accuracy, eps_all_losses)

# ----- Test both models for all training times ------ #
for model_name in models:
    row_list = []
    for epoch in epochs:
        # Evaluate the model
        model = tf.keras.models.load_model(model_dir + model_name + '/' + model_name + '_' + str(epoch))
        [avg_loss, avg_accuracy, all_losses] = evaluate_model(model)

        #Add average statistics to the master list (row_list)
        row_list.append([epoch, *avg_loss, *avg_accuracy])

        # Saves the list of all losses for a model and training time to a csv file
        loss_df = pd.DataFrame(all_losses, index = [*[str(eps) for eps in epsilons]])
        loss_df = loss_df.transpose()
        loss_df.to_csv(data_output_dir + '/all_losses/' + model_name + '_' + str(epoch) + '.csv', index=False)
        print(model_name + '_' + str(epoch) + '.csv created!')

    # Saves row_list, which includes average values for all training times and epsilon values, to a csv file
    general_df = pd.DataFrame(row_list, columns=['Epoch', *['Loss_' + str(eps) for eps in epsilons], *['Accuracy_' + str(eps) for eps in epsilons]])
    general_df.to_csv(data_output_dir + '/general/' + model_name + '.csv', index=False)
    print(model_name + '.csv created!')