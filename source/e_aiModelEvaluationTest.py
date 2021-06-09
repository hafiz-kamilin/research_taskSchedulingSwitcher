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
# for recording the computation time and loop waiting
from time import perf_counter
# for creating duplicate
from copy import deepcopy
# load tensorflow as machine learning framework
import tensorflow as tf
# load uber-cool loading bar
from tqdm import tqdm
# for data processing
import pandas as pd
# interfacing with tflite model
import numpy as np
# reading/saving csv file
import csv

""" load the dataset into the memory """

# function to load the dataset from csv file to the memory as a dataframe and split it
def csvToMemory(labelledTestDatasetName):

    train_df = pd.read_csv(labelledTestDatasetName)
    # get the dataframe size
    size = len(train_df.index)

    # create a dataframe that contain only the training data
    x_original = train_df.drop(
        columns=[
            "o_octasortAndFit",
            "o_sortAndFit",
            "o_shortestJobNext",
            "o_firstComeFirstServe"
        ]
    )
    # create a dataframe that contain only the target column
    y = train_df.drop(
        columns=[
            "i_numberOfTasks",
            "i_computationDeadline",
            "i_numberOfResourceConstraints"
        ]
    )
    # normalize the value of training data
    x_normalized = (x_original - x_original.min()) / (x_original.max() - x_original.min())
    # create a dataframe that contain only the target data
    y0 = train_df[["o_octasortAndFit"]]
    y1 = train_df[["o_sortAndFit"]]
    y2 = train_df[["o_shortestJobNext"]]
    y3 = train_df[["o_firstComeFirstServe"]]

    x_original = x_original.values.tolist()

    return x_original, x_normalized, y, y0, y1, y2, y3

""" load the already trained ai model """

# function to load the already trained ai model
def loadTrainedAIModel(AImodelName):

    # load the model
    model = tf.keras.models.load_model(AImodelName)

    return model

""" use .h5 model to evaluate the model """

# self made function to evaluate the .h5 model accuracy
def evaluateModelAccuracyH5(x_original, x_normalized, y, model, predictedResultName):

    # if prediction output folder does not exist
    if not os.path.exists("03_predict"):

        # create the model folder
        os.makedirs("03_predict")

    # open/create the csv file
    file = open(predictedResultName + "_h5.csv", "w", newline="")
    # set the class to write the file
    writer = csv.writer(file)
    # write the table's header to the csv file
    writer.writerow(
        [
            "i_numberOfResourceConstraints",
            "i_numberOfTasks",
            "i_computationDeadline",
            "o_octasortAndFit",
            "o_sortAndFit",
            "o_shortestJobNext",
            "o_firstComeFirstServe",
            "p_octasortAndFit",
            "p_sortAndFit",
            "p_shortestJobNext",
            "p_firstComeFirstServe"
        ]
    )

    # for storing the accuracy
    accuracy = 0
    # for storing the average computation time
    interfacingLatencyH5 = 0
    # get the dataset size
    datasetSize = x_normalized.shape[0]
    # convert the dataframe into a list
    y = y.values.tolist()

    # define the tqdm progress bar
    progress = tqdm(total=datasetSize)

    # get the prediction result in a batch that contain 1000 prediction
    for i in range(int(datasetSize)):

        # start the timer
        timer = perf_counter()

        # convert the values into something that tensorflow can process
        x = tf.Variable(
            [
                np.array(
                    [
                        x_normalized[["i_numberOfResourceConstraints"][0]][i],
                        x_normalized[["i_numberOfTasks"][0]][i],
                        x_normalized[["i_computationDeadline"][0]][i]
                    ]
                )
            ],
            trainable=True,
            dtype=tf.float64
        )

        # pass the parameters to the model and get the answer
        choosen = model(x)

        # stop the timer
        interfacingLatencyH5 += perf_counter() - timer

        highestValue = max(choosen)

        # rescaling the output as 0 and 1
        for j in range(len(choosen)):

            # swap the value with the highest probability as 1
            if (choosen[j] == highestValue):

                choosen[j] = 1

            # else is 0
            else:

                choosen[j] = 0

        # check if the prediction is accurate or not
        if (choosen == y[i]):

            accuracy += 1

        # update the tqdm progress bar
        progress.update(1)

        # write the result to the csv file
        writer.writerow(
            [
                x_original[i][0],
                x_original[i][1],
                x_original[i][2],
                choosen[0],
                choosen[1],
                choosen[2],
                choosen[3],
                y[i][0],
                y[i][1],
                y[i][2],
                y[i][3]
            ]
        )

    # recalculate the accuracy in percentage
    accuracy = accuracy / datasetSize * 100
    # get the average of the computation time
    interfacingLatencyH5 = interfacingLatencyH5 / datasetSize
    # close the tqdm progress bar
    progress.close()

    return accuracy, interfacingLatencyH5

# self made function to evaluate the .tflite model accuracy
def evaluateModelAccuracyTFLITE(x_original, x_normalized, y, AImodelName, predictedResultName):

    # if prediction output folder does not exist
    if not os.path.exists("03_predict"):

        # create the model folder
        os.makedirs("03_predict")

    # open/create the csv file
    file = open(predictedResultName + "_tflite.csv", "w", newline="")
    # set the class to write the file
    writer = csv.writer(file)
    # write the table's header to the csv file
    writer.writerow(
        [
            "i_numberOfResourceConstraints",
            "i_numberOfTasks",
            "i_computationDeadline",
            "o_octasortAndFit",
            "o_sortAndFit",
            "o_shortestJobNext",
            "o_firstComeFirstServe",
            "p_octasortAndFit",
            "p_sortAndFit",
            "p_shortestJobNext",
            "p_firstComeFirstServe"
        ]
    )

    # load tf lite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=AImodelName)
    interpreter.allocate_tensors()
    # get input and output tensors
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()
    # get the input shape of the model
    inputShape = inputDetails[0]["shape"]

    # for storing the accuracy
    accuracy = 0
    # for storing the average computation time
    interfacingLatencyTFLITE = 0
    # get the dataset size
    datasetSize = x_normalized.shape[0]
    # convert the dataframe into a list
    y = y.values.tolist()

    # define the tqdm progress bar
    progress = tqdm(total=datasetSize)

    # get the prediction result in a batch that contain 1000 prediction
    for i in range(int(datasetSize)):

        # start the timer
        timer = perf_counter()

        # convert the values into something that tensorflow lite can process
        inputData = np.array(
            [
                x_normalized[["i_numberOfResourceConstraints"][0]][i],
                x_normalized[["i_numberOfTasks"][0]][i],
                x_normalized[["i_computationDeadline"][0]][i]
            ]
        ).astype("float32").reshape(inputShape)

        # prepping the input data to be feed into the model
        interpreter.set_tensor(inputDetails[0]["index"], inputData)
        # process the input and get the output
        interpreter.invoke()
        choosen = [
            interpreter.get_tensor(outputDetails[0]["index"]),
            interpreter.get_tensor(outputDetails[1]["index"]),
            interpreter.get_tensor(outputDetails[2]["index"]),
            interpreter.get_tensor(outputDetails[3]["index"])
        ]

        # stop the timer
        interfacingLatencyTFLITE += perf_counter() - timer

        highestValue = max(choosen)

        # rescaling the output as 0 and 1
        for j in range(len(choosen)):

            # swap the value with the highest probability as 1
            if (choosen[j] == highestValue):

                choosen[j] = 1

            # else is 0
            else:

                choosen[j] = 0

        # check if the prediction is accurate or not
        if (choosen == y[i]):

            accuracy += 1

        # update the tqdm progress bar
        progress.update(1)

        # write the result to the csv file
        writer.writerow(
            [
                x_original[i][0],
                x_original[i][1],
                x_original[i][2],
                choosen[0],
                choosen[1],
                choosen[2],
                choosen[3],
                y[i][0],
                y[i][1],
                y[i][2],
                y[i][3]
            ]
        )

    # recalculate the accuracy in percentage
    accuracy = accuracy / datasetSize * 100
    # get the average of the computation time
    interfacingLatencyTFLITE = interfacingLatencyTFLITE / datasetSize
    # close the tqdm progress bar
    progress.close()

    return accuracy, interfacingLatencyTFLITE

""" use default method to evaluate the model """

# using a built in tensorflow function to evaluate the accuracy
def evaluateModelDefaultMethod(x_normalized, y0, y1, y2, y3, model):

    # evaluate the model on the test data using 'tf.evaluate' function
    # NOTE: the batch_size don't affect the loss and accuracy result
    _, _, _, _, _, _, _, _, _ = model.evaluate(x_normalized, [y0, y1, y2, y3], batch_size=1000)

# initiator
if __name__ == "__main__":

    os.system("cls")
    # dataset's filename
    labelledDatasetName = input("\nEnter the labelled dataset's name you want to test in 01_dataset folder (.csv): ")
    labelledTestDatasetName = "01_dataset/" + labelledDatasetName
    # model's filename
    AImodelNameH5 = input("Enter the trained model's name you want to test in 02_model folder (.h5): ")
    AImodelNameH5 = "02_model/" + AImodelNameH5
    # model's filename
    AImodelNameTFLITE = input("Enter the trained model's name you want to test in 02_model folder (.tflite): ")
    AImodelNameTFLITE = "02_model/" + AImodelNameTFLITE
    # prediction result filename
    predictedResultName = input("Enter the trained model's filename to be saved in 03_predict folder (don't add the .csv extension!): ")
    predictedResultName = "03_predict/" + predictedResultName

    print("\nLoad the test data into the memory.")

    # load a new test data into the memory
    x_original, x_normalized, y, y0, y1, y2, y3 = csvToMemory(labelledTestDatasetName)

    print("Load the trained AI model.")
    # load the already trained model
    model = loadTrainedAIModel(AImodelNameH5)

    print("\nCompute the AI accuracy using custom function and save the prediction output to the 03_predict folder.")

    accuracyH5, interfacingLatencyH5 = evaluateModelAccuracyH5(x_original, x_normalized, y, model, predictedResultName)
    print("The .h5 AI accuracy is " + str(accuracyH5) + "%")
    print("The .h5 interfacing latency is " + str(interfacingLatencyH5) + "s")
    accuracyTFLITE, interfacingLatencyTFLITE = evaluateModelAccuracyTFLITE(x_original, x_normalized, y, AImodelNameTFLITE, predictedResultName)
    print("The average .tflite AI accuracy is " + str(accuracyTFLITE) + "%")
    print("The average .tflite interfacing latency is " + str(interfacingLatencyTFLITE) + "s")

    # test the already trained model (only work on .h5 model)
    print("\nCompute the AI accuracy using default function.")
    evaluateModelDefaultMethod(x_normalized, y0, y1, y2, y3, model)
