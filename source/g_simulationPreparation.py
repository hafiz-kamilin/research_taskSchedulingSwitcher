#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

# load os feature module
import os
# supress tensorflow debug message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# load the machine learning framework
import tensorflow as tf

# share variables that has the similar values
import a_sharedVariableSetting as sVS
# reuse the codes to create the unlabeled dataset
import b_unlabeledDatasetGenerator as uDG
# reuse the code to create the labeled dataset
import c_trainingDatasetLabeler as tDL
# reuse the code to train AI model
import d_trainSchedulingSwitcher as tSS

# random number generation module
from random import uniform, seed, randint, shuffle
# for recording the computation time and loop waiting
from time import perf_counter, sleep
# to color warning text with red color
from colorama import Fore, Style
# duplicate variable into a new memory
from copy import deepcopy
# load uber-cool loading bar
from tqdm import tqdm
# for finding specific files
from glob import glob
# reading data from pickle
import pickle

""" rewrite the device unit based on the number of people inside the building and create the tasks to be scheduled """

# function that rewrite the number active device unit according to the number of people
def deviceUnitMultiplier(typeOfDevices, peopleNum):

    # for storing the task to be scheduled
    taskToBeScheduled = []

    # NOTE: using ternary operators / conditional expression
    activatedDevices = {
        "airConditioner0": [
            10 if (peopleNum >= 40) else
            8 if (peopleNum >= 30 and peopleNum < 40) else
            6 if (peopleNum >= 20 and peopleNum < 30) else
            4 if (peopleNum >= 10 and peopleNum < 20) else
            2 if (peopleNum >= 5 and peopleNum < 10) else
            1,
            typeOfDevices["airConditioner0"][1]
        ],
        "temperatureSensor0": [
            10 if (peopleNum >= 40) else
            8 if (peopleNum >= 30 and peopleNum < 40) else
            6 if (peopleNum >= 20 and peopleNum < 30) else
            4 if (peopleNum >= 10 and peopleNum < 20) else
            2 if (peopleNum >= 5 and peopleNum < 10) else
            1,
            typeOfDevices["temperatureSensor0"][1]
        ],
        "motionSensor0": [
            8 if (peopleNum >= 40) else
            6 if (peopleNum >= 30 and peopleNum < 40) else
            4 if (peopleNum >= 20 and peopleNum < 30) else
            2 if (peopleNum >= 10 and peopleNum < 20) else
            1,
            typeOfDevices["motionSensor0"][1]
        ],
        "smokeSensor0": [
            13 if (peopleNum >= 30) else
            7 if (peopleNum >= 10 and peopleNum < 30) else
            4,
            typeOfDevices["smokeSensor0"][1]
        ],
        "securityDoor0": [
            8 if (peopleNum >= 40) else
            6 if (peopleNum >= 30 and peopleNum < 40) else
            4 if (peopleNum >= 20 and peopleNum < 30) else
            2 if (peopleNum >= 10 and peopleNum < 20) else
            1,
            typeOfDevices["smokeSensor0"][1]
        ],
        "securityCamera0": [
            6 if (peopleNum >= 30) else
            4 if (peopleNum >= 10 and peopleNum < 30) else
            2,
            typeOfDevices["securityCamera0"][1]
        ],
        "lighting0": [
            40 if (peopleNum >= 40) else
            30 if (peopleNum >= 30 and peopleNum < 40) else
            20 if (peopleNum >= 20 and peopleNum < 30) else
            10 if (peopleNum >= 10 and peopleNum < 20) else
            5 if (peopleNum >= 5 and peopleNum < 10) else
            0,
            typeOfDevices["lighting0"][1]
        ],
        "lighting1": [
            1 if (peopleNum >= 40) else
            2 if (peopleNum >= 30 and peopleNum < 40) else
            3 if (peopleNum >= 20 and peopleNum < 30) else
            4 if (peopleNum >= 10 and peopleNum < 20) else
            5,
            typeOfDevices["lighting0"][1]
        ]
    }

    for device in activatedDevices.keys():

        # randomized the number of device's unit to be scheduled
        numberOfTasks = randint(0, activatedDevices[device][0])

        for i in range(numberOfTasks):

            # randomize the iot task runtime based on the mean value of the resource constraint
            runtime = round(uniform(1, sum(activatedDevices[device][1].values()) / len(activatedDevices[device][1])), 3)
            # save the ranomized task
            taskToBeScheduled.append([device + "_" + str(i), runtime, activatedDevices[device][1]])

    # shuffle the sequence of tasks to be scheduled
    shuffle(taskToBeScheduled)
    shuffle(taskToBeScheduled)
    shuffle(taskToBeScheduled)

    return taskToBeScheduled

""" record the algorithm computation time """

# function to automate the scheduling of the tasks via FCFS, SJF, SnF, OnF
def algorithmComputationTime(resourceConstraints, taskToBeScheduled):

    computationTime = {}

    # start the timer
    timer0 = perf_counter()
    # compute the scheduler by using FCFS
    _, schedulerDuration = uDG.firstComeFirstServe(resourceConstraints, taskToBeScheduled)
    # stop the timer
    timer0 = perf_counter() - timer0

    computationTime["firstComeFirstServe"] = {"computationTime": timer0, "schedulerDuration": schedulerDuration}

    # start the timer
    timer1 = perf_counter()
    # compute the scheduler by using SJF (actually shortest job next can be called as shortest job first)
    _, schedulerDuration = uDG.shortestJobNext(resourceConstraints, taskToBeScheduled)
    # stop the timer
    timer1 = perf_counter() - timer1

    computationTime["shortestJobNext"] = {"computationTime": timer1, "schedulerDuration": schedulerDuration}

    # start the timer
    timer2 = perf_counter()
    # compute the scheduler by using SnF
    _, schedulerDuration = uDG.sortAndFit(resourceConstraints, taskToBeScheduled)
    # stop the timer
    timer2 = perf_counter() - timer2

    computationTime["sortAndFit"] = {"computationTime": timer2, "schedulerDuration": schedulerDuration}

    # start the timer
    timer3 = perf_counter()
    # compute the scheduler by using OnF
    _, schedulerDuration = uDG.octasortAndFit(resourceConstraints, taskToBeScheduled)
    # stop the timer
    timer3 = perf_counter() - timer3

    computationTime["octasortAndFit"] = {"computationTime": timer3, "schedulerDuration": schedulerDuration}

    return computationTime

""" creating a specific dataset to train the MTL model """

# function to create a dataset to train a new task scheduling switcher model
def trainingDatasetCreator(folderName, maxPeople, typeOfDevices, numberOfData, resourceConstraints, seedKey):

    unlabeledDataset = {}
    labeledDataset = {}
    print("\nGenerating the unlabeled dataset to train the Task Scheduling Switcher Framework.")
    # define the tqdm progress bar
    progress = tqdm(total=numberOfData)

    currentPeopleNum = 0

    for data in range(numberOfData):

        if (currentPeopleNum == maxPeople):

            currentPeopleNum = 0

        currentPeopleNum += 1

        # generate the task to be scheduled
        taskToBeScheduled = deviceUnitMultiplier(
            typeOfDevices, currentPeopleNum
        )

        # get the computation time
        computationTime = algorithmComputationTime(resourceConstraints, taskToBeScheduled)

        unlabeledDataset[data] = {
            "resourceConstraints": len(resourceConstraints),
            "taskToBeScheduled": len(taskToBeScheduled),
            "schedulingMethod": {
                "octasortAndFit": {
                    "computationTime": computationTime["octasortAndFit"]["computationTime"],
                    "schedulerDuration": computationTime["octasortAndFit"]["schedulerDuration"]
                },
                "sortAndFit": {
                    "computationTime": computationTime["sortAndFit"]["computationTime"],
                    "schedulerDuration": computationTime["sortAndFit"]["schedulerDuration"]
                },
                "shortestJobNext": {
                    "computationTime": computationTime["shortestJobNext"]["computationTime"],
                    "schedulerDuration": computationTime["shortestJobNext"]["schedulerDuration"]
                },
                "firstComeFirstServe": {
                    "computationTime": computationTime["firstComeFirstServe"]["computationTime"],
                    "schedulerDuration": computationTime["firstComeFirstServe"]["schedulerDuration"]
                }
            }
        }

        # update the tqdm progress bar
        progress.update(1)

    # close the tqdm progress bar
    progress.close()

    # create and open the file
    pickleOut = open(folderName + "unlabeledDataset.pickle", "wb")
    # write the unlabeledDataset into the file
    pickle.dump(unlabeledDataset, pickleOut)
    # close the file
    pickleOut.close()

    tDL.datasetWriter(
        seedKey,
        None,
        unlabeledDataset,
        folderName + "labeledDataset.csv"
    )

# function to train the model based on previously constructed MTL model
def trainNewAIModel(folderName, seedKey):

    # load the dataset into the memory and prepare it for training
    train_X, test_X, train_y0, test_y0, train_y1, test_y1, train_y2, test_y2, train_y3, test_y3, input_columns = tSS.csvToMemory(
        folderName + "labeledDataset.csv",
        0.7,
        seedKey
    )

    # train the AI model and return the training history
    history = tSS.trainAIModel(
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
        folderName
    )

    # list all .h5 file exist in folderName folder
    h5FilesFound = [file
                    for path, subdir, files in os.walk(folderName)
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

    # accuracy and loss filename
    accuracyName = folderName + "accuracy(" + str(seedKey) + ").png"
    lossName = folderName + "loss(" + str(seedKey) + ").png"
    # plotted data for the training history filename
    trainingHistoryName = folderName + "trainingHistory(" + str(seedKey) + ").csv"
    # plot the result
    tSS.plotResult(history, accuracyName, lossName, trainingHistoryName)

# initiator
if __name__ == "__main__":

    folderName = "./04_simulate/"
    # if dataset folder does not exist
    if not os.path.exists(folderName):

        # create the dataset folder
        os.makedirs(folderName)

    os.system("cls")
    # because it will affect the dataset creation time measurement
    print(Fore.RED + "\n1. Make sure that you are launching this simulation program without any background tasks running!")
    print(Fore.RED + "2. The simulation program must run on the same computer where the training datasets were created!" + Style.RESET_ALL)
    # randomization key to make the dataset reproducible
    seedKey = int(input("Enter the randomization key: "))

    # make the randomization reproducable
    seed(seedKey)

    # NOTE: here we hard-coded the number of people that present inside the building
    #       where the dictionary keys represent hours in day to be used in the simulation
    # type of devices used in this simulation
    numberOfPeople = {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 6,
        7: 27,
        8: 43,
        9: 50,
        10: 50,
        11: 50,
        12: 50,
        13: 34,
        14: 45,
        15: 50,
        16: 50,
        17: 41,
        18: 32,
        19: 20,
        20: 15,
        21: 10,
        22: 5,
        23: 2
    }

    # NOTE: here we hard-coded the type of devices and it number to be used in the simulation
    typeOfDevices = {
        # deviceID: [numberOfUnit, {0: resource0Taken, 1: resource1Taken, 2: resource2Taken, ...}]
        "airConditioner0": [10, {0: 400, 1: 12, 2: 12}],
        "temperatureSensor0": [10, {0: 15, 1: 4, 2: 6}],
        "motionSensor0": [8, {0: 20, 1: 6, 2: 8}],
        "smokeSensor0": [13, {0: 16, 1: 10, 2: 15}],
        "securityDoor0": [8, {0: 100, 1: 3, 2: 5}],
        "securityCamera0": [6, {0: 100, 1: 10, 2: 12}],
        "lighting0": [40, {0: 20, 1: 1, 2: 1}],
        "lighting1": [5, {0: 10, 1: 1, 2: 1}],
    }

    # NOTE: here we hard-coded the resourceConstraints to be used in the simulation
    # 5000W, 50MB/s and 100MB/s
    resourceConstraints = {0: 5000, 1: 50, 2: 100}

    # generate the labeled dataset and save it as a csv file
    trainingDatasetCreator(folderName, 50, typeOfDevices, sVS.cycle, resourceConstraints, seedKey)
    # train a new model based on the previously constructed MTL model
    trainNewAIModel(folderName, seedKey)
