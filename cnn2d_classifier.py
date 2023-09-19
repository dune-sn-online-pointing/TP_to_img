import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-input_data", help="input data file", default="dataset_img.npy", type=str)
parser.add_argument("-input_label", help="input label file", default="dataset_lab.npy", type=str)
parser.add_argument("-input_path", help="input path", default="/eos/user/d/dapullia/tp_dataset/", type=str)
parser.add_argument("-save_path", help="save path", default="/eos/user/d/dapullia/tp_dataset/", type=str)
parser.add_argument("-model_name", help="model name", default="model.h5", type=str)
parser.add_argument('-load_model', action='store_true', help='save the model')


args = parser.parse_args()

input_data = args.input_data
input_label = args.input_label
input_path = args.input_path
save_path = args.save_path
model_name = args.model_name
load_model = args.load_model

if not os.path.exists(input_path+input_data) or not os.path.exists(input_path+input_label):
    print("Input file not found.")
    exit()

if not os.path.exists(save_path):
    os.makedirs(save_path)




'''
Image shape: (, 1000, 70)
'''

# Check if GPU is available
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# Load the dataset
print("Loading the dataset...")
dataset_img = np.load(input_path+input_data)
dataset_label = np.load(input_path+input_label)
print("Dataset loaded.")
print("Dataset_img shape: ", dataset_img.shape)
print("Dataset_lab shape: ", dataset_label.shape)

# Check if the dimension of images and labels are the same
if dataset_img.shape[0] != dataset_label.shape[0]:
    print("Error: the dimension of images and labels are not the same.")
    exit()


# Remove the images with label 10
print("Different labels: ", np.unique(dataset_label, return_counts=True))


print("Removing the images with label 10...")
print("Dataset_img shape before: ", dataset_img.shape)
print("Dataset_lab shape before: ", dataset_label.shape)
index = np.where(dataset_label == 10)
print("Index: ", index)
dataset_img = np.delete(dataset_img, index, axis=0)
dataset_label = np.delete(dataset_label, index, axis=0)

print("Dataset_img shape after: ", dataset_img.shape)
print("Dataset_lab shape after: ", dataset_label.shape)
print("Images with label 10 removed.")

# shuffle the dataset
print("Shuffling the dataset...")
index = np.arange(dataset_img.shape[0])
np.random.shuffle(index)
dataset_img = dataset_img[index]
dataset_label = dataset_label[index]
print("Dataset shuffled.")

# Split the dataset in training and test
print("Splitting the dataset...")

split = 0.8

train_images = dataset_img[:int(dataset_img.shape[0]*split)]
test_images = dataset_img[int(dataset_img.shape[0]*split):]
train_labels = dataset_label[:int(dataset_label.shape[0]*split)]
test_labels = dataset_label[int(dataset_label.shape[0]*split):]

print("Dataset splitted.")

# create 1 hot encoding for the labels
print("Creating 1 hot encoding for the labels...")
print("Train shape before: ",train_labels.shape)
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
print("Train shape after: ",train_labels.shape)
print("1 hot encoding created.")

# show images where the label is 1 and 4

print(train_labels.shape)

if not os.path.exists(save_path+'samples/'):
    os.makedirs(save_path+'samples/')

c=0
for i in range(len(train_labels)):
    if train_labels[i][1] == 1:
        plt.imshow(train_images[i])
        plt.savefig(save_path+ "samples/1_"+str(c)+".png")
        c+=1
    if c==10:
        break

c=0
for i in range(len(train_labels)):
    if train_labels[i][4] == 1:
        plt.imshow(train_images[i])
        plt.savefig(save_path+ "samples/4_"+str(c)+".png")
        c+=1
    if c==10:
        break


if not load_model or not os.path.exists(save_path+model_name):
    if not os.path.exists(save_path+model_name):
        print("Model not found, building a new one...")
    # Build the model, cnn 2D
    print("Building the model...")

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (30, 3), activation='relu', input_shape=(1000, 70, 1)))
    # shape: (1000, 70, 32)
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((5, 2)))
    # shape: (200, 35, 32)
    model.add(layers.Conv2D(64, (30, 3), activation='relu'))
    # shape: (171, 33, 64)
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((5, 2)))
    # shape: (34, 16, 64)
    model.add(layers.Conv2D(128, (30, 3), activation='relu'))
    # shape: (5, 14, 128)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dense(64, activation='linear'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dense(10, activation='softmax'))  


    # Compile the model
    print("Compiling the model...")
    # add learning ratescheduler

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1,
        decay_steps=10000,
        decay_rate=0.96)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])   



    print("Model compiled.")

    print("Model built.")

    # Train the model
    print("Training the model...")

    callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=20,
            verbose=1)
    ]


    history = model.fit(train_images, train_labels, epochs=500, batch_size=32, validation_data=(test_images, test_labels), callbacks=callbacks)
    print("Model trained.")

    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=32)
    print("Model evaluated.")
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    # Plot the training and validation loss
    print("Plotting the training and validation loss...")
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)
    plt.figure()
    plt.plot(epochs, loss_values, 'bo', label='Training loss')  # bo = blue dot
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # b = "solid blue line"
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path+model_name+"_loss.png")
    plt.figure()
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')  # bo = blue dot
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')  # b = "solid blue line"
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path+model_name+"_accuracy.png")
    print("Plot saved.")

    # Save the model
    print("Saving the model...")
    model.save(save_path+model_name)
    print("Model saved.")


else:
    # Load the model
    model = keras.models.load_model(save_path+model_name)


# Do some test

print("Doing some test...")
predictions = model.predict(test_images)
print("Predictions: ", predictions)
print("Test labels: ", test_labels)

# Evaluate the ROC curve
print("Evaluating the ROC curve...")


# Binarize the output
y_test = label_binarize(test_labels, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y_test.shape[1]

print(y_test.shape)
print(predictions.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()

plt.figure()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig(save_path+"ROC_curve.png")
print("ROC curve evaluated and saved.")




