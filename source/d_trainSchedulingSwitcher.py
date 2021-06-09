#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

# load os feature module
import os
# supress tensorflow debug message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# for plotting graph
import matplotlib.pyplot as plt
# load the machine learning framework
import tensorflow as tf
# for finding specific files
from glob import glob
# for data processing
import pandas as pd
# reading/saving csv file
import csv

# share variables that has the similar values
import a_sharedVariableSetting as sVS

""" load the dataset into the memory and split it into training and testing """

# function to load the dataset from csv file to the memory as a dataframe and split it
def csvToMemory(labelledDatasetName, split, seedKey):

    df = pd.read_csv(labelledDatasetName)

    # create a dataframe that contain only the training data
    x = df.drop(
        columns=[
            "o_octasortAndFit",
            "o_sortAndFit",
            "o_shortestJobNext",
            "o_firstComeFirstServe"
        ]
    )
    # create a dataframe that contain only the target data
    y0 = df[["o_octasortAndFit"]]
    y1 = df[["o_sortAndFit"]]
    y2 = df[["o_shortestJobNext"]]
    y3 = df[["o_firstComeFirstServe"]]

    # normalize the value of training data
    x["i_numberOfResourceConstraints"] = x["i_numberOfResourceConstraints"].apply(
        lambda col: (
            col - sVS.numberOfResourceConstraints_range[0]
        ) / (
            sVS.numberOfResourceConstraints_range[1] - sVS.numberOfResourceConstraints_range[0]
        )
    )
    x["i_numberOfTasks"] = x["i_numberOfTasks"].apply(
        lambda col: (
            col - sVS.numberOfTasks_range[0]
        ) / (
            sVS.numberOfTasks_range[1] - sVS.numberOfTasks_range[0]
        )
    )
    x["i_computationDeadline"] = x["i_computationDeadline"].apply(
        lambda col: (
            col - 0
        ) / (
            sVS.maxComputationDeadline - 0
        )
    )

    # split the x dataset into training and test
    train_X = x.copy().sample(frac=split, random_state=seedKey)
    test_X = x.copy().sample(frac=1 - split, random_state=seedKey)
    # split the y dataset into training and test
    train_y0 = y0.copy().sample(frac=split, random_state=seedKey)
    test_y0 = y0.copy().sample(frac=1 - split, random_state=seedKey)
    train_y1 = y1.copy().sample(frac=split, random_state=seedKey)
    test_y1 = y1.copy().sample(frac=1 - split, random_state=seedKey)
    train_y2 = y2.copy().sample(frac=split, random_state=seedKey)
    test_y2 = y2.copy().sample(frac=1 - split, random_state=seedKey)
    train_y3 = y3.copy().sample(frac=split, random_state=seedKey)
    test_y3 = y3.copy().sample(frac=1 - split, random_state=seedKey)

    # get number of columns in training data
    input_columns = train_X.shape[1]

    return train_X, test_X, train_y0, test_y0, train_y1, test_y1, train_y2, test_y2, train_y3, test_y3, input_columns

""" train the ai based on the splitted dataset """

# function to train the AI
def trainAIModel(seedKey, train_X, test_X, train_y0, test_y0, train_y1, test_y1, train_y2, test_y2, train_y3, test_y3, input_columns, modelName):

    # if model folder does not exist
    if not os.path.exists("02_model"):

        # create the model folder
        os.makedirs("02_model")

    # lock the training randomization
    tf.random.set_seed(
        seedKey
    )

    # define the number of epoch
    # NOTE: because of .h5 file deleting mechanism in main function and the naming scheme in
    #       ModelCheckpoint, epoch number must not exceed 1000 otherwise model with epoch number
    #       10000 above will be deleted first!
    epochNum = 3000
    # define the batch size
    batchSIze = 1000
    # define the model's input
    input = tf.keras.layers.Input(shape=(input_columns,))
    # define the layer 1 for the hard parameter sharing
    hardSharing1 = tf.keras.layers.Dense(input_columns)(input)
    # define the layer 2 for the hard parameter sharing
    hardSharing2 = tf.keras.layers.Dense(input_columns)(hardSharing1)
    # define the layer 3 for the hard parameter sharing
    hardSharing3 = tf.keras.layers.Dense(input_columns)(hardSharing2)
    # define the model's output 0 (o_octasortAndFit)
    hiddenLayer0_0 = tf.keras.layers.Dense(3, activation="relu")(hardSharing3)
    hiddenLayer0_1 = tf.keras.layers.Dense(2, activation="relu")(hiddenLayer0_0)
    output0 = tf.keras.layers.Dense(1, activation="sigmoid", name="0")(hiddenLayer0_1)
    # define the model's output 1 (o_sortAndFit)
    hiddenLayer1_0 = tf.keras.layers.Dense(3, activation="relu")(hardSharing3)
    hiddenLayer1_1 = tf.keras.layers.Dense(2, activation="relu")(hiddenLayer1_0)
    output1 = tf.keras.layers.Dense(1, activation="sigmoid", name="1")(hiddenLayer1_1)
    # define the model's output 2 (o_shortestJobNext)
    hiddenLayer2_0 = tf.keras.layers.Dense(3, activation="relu")(hardSharing3)
    hiddenLayer2_1 = tf.keras.layers.Dense(2, activation="relu")(hiddenLayer2_0)
    output2 = tf.keras.layers.Dense(1, activation="sigmoid", name="2")(hiddenLayer2_1)
    # define the model's output 3 (o_firstComeFirstServe)
    hiddenLayer3_0 = tf.keras.layers.Dense(3, activation="relu")(hardSharing3)
    hiddenLayer3_1 = tf.keras.layers.Dense(2, activation="relu")(hiddenLayer3_0)
    output3 = tf.keras.layers.Dense(1, activation="sigmoid", name="3")(hiddenLayer3_1)

    # define the combined losses parameter for each classification output
    multitaskLoss = {
        "0": tf.keras.losses.binary_crossentropy,
        "1": tf.keras.losses.binary_crossentropy,
        "2": tf.keras.losses.binary_crossentropy,
        "3": tf.keras.losses.binary_crossentropy,
    }
    multitaskLossWeights = {
        "0": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
    }

    # define the input and output for the model
    model = tf.keras.models.Model(
        inputs=input,
        outputs=[
            output0,
            output1,
            output2,
            output3
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.001 / epochNum)
    # compile the model using mse as a measure of model performance
    model.compile(optimizer=opt, loss=multitaskLoss, loss_weights=multitaskLossWeights, metrics=["accuracy"])
    # set the file naming based on the loss and accuracy value
    a = "{val_0_loss:.2f},{val_1_loss:.2f},{val_2_loss:.2f},{val_3_loss:.2f}"
    b = "{val_0_accuracy:.2f},{val_1_accuracy:.2f},{val_2_accuracy:.2f},{val_3_accuracy:.2f}"
    # operation that need to be done after each epoch
    postEpoch = [
        # define an auto saving feature where it will continuously save the model with the lowest val_loss value
        tf.keras.callbacks.ModelCheckpoint(
            filepath=modelName + "{epoch:04d}_valLoss=" + a + "_valAcc=" + b + ".h5",
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            save_best_only=True
        ),
        # define an early stopping feature to stop the training when the model could no longer improve the val_loss value
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto"
        )
    ]

    # train the model now
    history = model.fit(
        train_X,
        [
            train_y0,
            train_y1,
            train_y2,
            train_y3
        ],
        batch_size=batchSIze,
        epochs=epochNum,
        verbose=2,
        callbacks=postEpoch,
        validation_data=(
            test_X,
            [
                test_y0,
                test_y1,
                test_y2,
                test_y3
            ],
        ),
        shuffle=True
    )

    return history

""" plot the training history into a graph """

# function to plot and display the training results (accuracy and loss)
def plotResult(history, accuracyName, lossName, trainingHistoryName):

    # open/create the csv file
    file = open(trainingHistoryName, "w", newline="")
    # set the class to write the file
    writer = csv.writer(file)
    # write the table's header to the csv file
    writer.writerow(
        [
            "epoch",
            "loss",
            "0_loss",
            "1_loss",
            "2_loss",
            "3_loss",
            "0_accuracy",
            "1_accuracy",
            "2_accuracy",
            "3_accuracy",
            "val_loss",
            "val_0_loss",
            "val_1_loss",
            "val_2_loss",
            "val_3_loss",
            "val_0_accuracy",
            "val_1_accuracy",
            "val_2_accuracy",
            "val_3_accuracy",
        ]
    )

    # write the training history to the csv file
    for i in range(len(history.epoch)):

        # one by one
        writer.writerow(
            [
                history.epoch[i],
                history.history["loss"][i],
                history.history["0_loss"][i],
                history.history["1_loss"][i],
                history.history["2_loss"][i],
                history.history["3_loss"][i],
                history.history["0_accuracy"][i],
                history.history["1_accuracy"][i],
                history.history["2_accuracy"][i],
                history.history["3_accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_0_loss"][i],
                history.history["val_1_loss"][i],
                history.history["val_2_loss"][i],
                history.history["val_3_loss"][i],
                history.history["val_0_accuracy"][i],
                history.history["val_1_accuracy"][i],
                history.history["val_2_accuracy"][i],
                history.history["val_3_accuracy"][i]
            ]
        )

    # plot the training and test accuracy
    plt.figure(1)
    plt.plot(history.history["0_accuracy"])
    plt.plot(history.history["1_accuracy"])
    plt.plot(history.history["2_accuracy"])
    plt.plot(history.history["3_accuracy"])
    plt.plot(history.history["val_0_accuracy"])
    plt.plot(history.history["val_1_accuracy"])
    plt.plot(history.history["val_2_accuracy"])
    plt.plot(history.history["val_3_accuracy"])
    plt.title("AI Scheduling Switcher's Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0.0, 1.0)
    plt.legend(["train0", "train1", "train2", "train3", "test0", "test1", "test2", "test3"], loc='lower left')
    plt.grid()
    plt.savefig(accuracyName)
    plt.show()

    # plot the training and test loss
    plt.figure(2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["0_loss"])
    plt.plot(history.history["1_loss"])
    plt.plot(history.history["2_loss"])
    plt.plot(history.history["3_loss"])
    plt.plot(history.history["val_loss"])
    plt.plot(history.history["val_0_loss"])
    plt.plot(history.history["val_1_loss"])
    plt.plot(history.history["val_2_loss"])
    plt.plot(history.history["val_3_loss"])
    plt.title("AI Scheduling Switcher's Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0.0, 2.0)
    plt.legend(["train", "train0", "train1", "train2", "train3", "test", "test0", "test1", "test2", "test3"], loc='upper left')
    plt.grid()
    plt.savefig(lossName)
    plt.show()

# initiator
if __name__ == "__main__":

    os.system("cls")
    # labelled dataset filename
    labelledDatasetName = input("\nEnter the labelled dataset's name you want to use in 01_dataset folder (.csv): ")
    labelledDatasetName = "01_dataset/" + labelledDatasetName
    # model's folder
    modelName = "02_model/"
    # randomization key
    seedKey = int(input("Enter the randomization key: "))
    # accuracy and loss filename
    accuracyName = "02_model/accuracy(" + str(seedKey) + ").png"
    lossName = "02_model/loss(" + str(seedKey) + ").png"
    # plotted data for the training history filename
    trainingHistoryName = "02_model/trainingHistory(" + str(seedKey) + ").csv"
    # define how many data will be used to train the model
    split = 0.7
    # location to save the trained data

    # load the dataset into the memory and prepare it for training
    train_X, test_X, train_y0, test_y0, train_y1, test_y1, train_y2, test_y2, train_y3, test_y3, input_columns = csvToMemory(labelledDatasetName, split, seedKey)

    print("\nTraining start now.\n")

    # train the AI model and return the training history
    history = trainAIModel(
        seedKey,
        train_X,
        test_X,
        train_y0,
        test_y0,
        train_y1,
        test_y1,
        train_y2,
        test_y2,
        train_y3,
        test_y3,
        input_columns,
        modelName
    )

    print("\nTraining completed.")

    # list all .h5 file exist in 02_model/ folder
    h5FilesFound = [file
                    for path, subdir, files in os.walk(modelName)
                    for file in glob(os.path.join(path, "*.h5"))]
    # sort the .h5 files according to the epoch number
    h5FilesFound = sorted(h5FilesFound)

    # delete all the .h5 files except for the last file that has the biggest epoch number
    for i in range(len(h5FilesFound) - 1):

        os.remove(h5FilesFound[i])

    # load the model that is not deleted to be quantize (model is a 32 bit floating point)
    model = tf.keras.models.load_model(h5FilesFound[-1])
    # prepping the model quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # target hardware is old gtx 1050ti gpu with 8 bit floating point (can also be targetted on a cpu)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # NOTE: if the target hardware is a latest gpu, uncomment the line below to convert the model to a 16 bit floating point
    #       converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    # saving converted model as .tflite file
    open(h5FilesFound[-1].replace("h5", "tflite"), "wb").write(tflite_quant_model)

    # plot the result
    plotResult(history, accuracyName, lossName, trainingHistoryName)
