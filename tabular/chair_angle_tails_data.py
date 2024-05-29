# Code for formatting the .pkl files for the ChairAngle_Tails dataset produced by this repo
# https://github.com/fregu856/regression_uncertainty/tree/main

import pickle

import numpy as np


def image_normalize(img_array):

    #Initial shape (N, 3, 64, 64). Expected: (N, 64, 64, 3)
    img_array = np.transpose(img_array, (0, 2, 3, 1)) # originally `transpose(img, (1, 2, 0))`

    img_array = img_array/255.0
    img_array = img_array - np.array([0.485, 0.456, 0.406])
    img_array = img_array/np.array([0.229, 0.224, 0.225])
    img_array = np.transpose(img_array, (0, 3, 1, 2)) # (shape: (3, 64, 64))
    img_array = img_array.astype(np.float32)

    return img_array


def load_data():

    # Load labels
    with open("./data/ChairAngle_Tails/labels_train.pkl", "rb") as file:
        train_labels = pickle.load(file)
    with open("./data/ChairAngle_Tails/labels_val.pkl", "rb") as file:
        val_labels = pickle.load(file)
    with open("./data/ChairAngle_Tails/labels_test.pkl", "rb") as file:
        test_labels = pickle.load(file)

    # Load images

    with open("./data/ChairAngle_Tails/images_train.pkl", "rb") as file:
        train_images = pickle.load(file)
    with open("./data/ChairAngle_Tails/images_val.pkl", "rb") as file:
        val_images = pickle.load(file)
    with open("./data/ChairAngle_Tails/images_test.pkl", "rb") as file:
        test_images = pickle.load(file)

    # Perform normalization
    train_images = image_normalize(train_images)
    val_images = image_normalize(val_images)
    test_images = image_normalize(test_images)

    # Pack up the data and return it

    data_packet = {
        'x_train': train_images,
        'x_valid': val_images,
        'x_test': test_images,
        'y_train': train_labels[:, None],
        'y_valid': val_labels[:, None],
        'y_test': test_labels[:, None],
    }

    return data_packet
