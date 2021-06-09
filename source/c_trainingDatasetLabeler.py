#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

# random number generation module
from random import uniform, seed
# load uber-cool loading bar
from tqdm import tqdm
# reading data from pickle
import pickle
# reading/saving csv file
import csv
# get os specific feature
import os

# share variables that has the similar values
import a_sharedVariableSetting as sVS

""" perform a controllable computation deadline computation """

# function to randomize the computation deadline
def randomizeDeadline(minimumComputationTime, maximumComputationTime):

    return uniform(minimumComputationTime, maximumComputationTime)

""" priming the raw dataset so that the AI can use it for training/validating """

# function to prime the dataset so that the AI can use it for training/validating
# NOTE: for the sake of reusability, i have separated the part of labeling and saving
#       the data into a csv file
def datasetLabeler(dataset, i):

    computationTimeForEachSchedulingAlgorithm = [
        [0, dataset[i]["schedulingMethod"]["octasortAndFit"]["computationTime"]],
        [1, dataset[i]["schedulingMethod"]["sortAndFit"]["computationTime"]],
        [2, dataset[i]["schedulingMethod"]["shortestJobNext"]["computationTime"]],
        [3, dataset[i]["schedulingMethod"]["firstComeFirstServe"]["computationTime"]]
    ]

    # sort the task in ascending order
    computationTimeForEachSchedulingAlgorithm = sorted(computationTimeForEachSchedulingAlgorithm, key=lambda x: x[1])
    # initialize a dictionary to store the computation deadline as key, and choosen scheduling algorithm as value
    generatedLabel = {}

    for j in range(len(computationTimeForEachSchedulingAlgorithm)):

        computationDeadline = 0

        # if it is not the last scheduling algorithm to be randomized the computation deadline
        if (j != (len(computationTimeForEachSchedulingAlgorithm) - 1)):

            # randomize the computation deadline
            computationDeadline = computationTimeForEachSchedulingAlgorithm[j][1] + randomizeDeadline(
                0,
                abs(computationTimeForEachSchedulingAlgorithm[j][1] - computationTimeForEachSchedulingAlgorithm[j + 1][1])
            )

        # it is the last to be randomized the deadline
        else:

            # randomize the computation deadline
            # NOTE: maximum computation deadline is 1 second
            computationDeadline = computationTimeForEachSchedulingAlgorithm[j][1] + randomizeDeadline(
                0,
                sVS.maxComputationDeadline - computationTimeForEachSchedulingAlgorithm[j][1]
            )

        # initialize the list that store which scheduling method are choosen
        choosenOrNot = []

        if (computationTimeForEachSchedulingAlgorithm[j][0] == 0):

            # prioritize octasortAndFit
            choosenOrNot = [1, 0, 0, 0]

        elif (computationTimeForEachSchedulingAlgorithm[j][0] == 1):

            # prioritize sortAndFit
            choosenOrNot = [0, 1, 0, 0]

        elif (computationTimeForEachSchedulingAlgorithm[j][0] == 2):

            # prioritize shortestJobNext
            choosenOrNot = [0, 0, 1, 0]

        elif (computationTimeForEachSchedulingAlgorithm[j][0] == 3):

            # prioritize firstComeFirstServe
            choosenOrNot = [0, 0, 0, 1]

        # save the choosen scheduler based on the randomized computation deadline
        generatedLabel[computationDeadline] = choosenOrNot

    return generatedLabel

# function to write a labeled dataset into the csv file
def datasetWriter(seedKey, unlabeledDatasetName, unlabeledDataset, labeledDatasetName):

    # make the randomization reproducable
    seed(seedKey)

    if (unlabeledDatasetName is not None):

        print("\nLoading the raw dataset into the memory.")

        # read and load the pickle file to the memory
        dataset = pickle.load(open(unlabeledDatasetName, "rb"))

    else:

        dataset = unlabeledDataset

    # find the dataset length
    cycle = len(dataset)

    # open/create the csv file
    file = open(labeledDatasetName, "w", newline="")
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
            "o_firstComeFirstServe"
        ]
    )

    print("\nLabeling up the raw dataset.")

    # define the tqdm progress bar
    progress = tqdm(total=cycle)

    for i in range(cycle):

        # label the dataset in a way each scheduling algorithm will have
        # an equal chance to meet the computation deadline
        generatedLabel = datasetLabeler(dataset, i)

        for computationDeadline in generatedLabel:

            # write the data to the csv file
            writer.writerow(
                [
                    dataset[i]["resourceConstraints"],
                    dataset[i]["taskToBeScheduled"],
                    computationDeadline,
                    generatedLabel[computationDeadline][0],
                    generatedLabel[computationDeadline][1],
                    generatedLabel[computationDeadline][2],
                    generatedLabel[computationDeadline][3]
                ]
            )

        # update the tqdm progress bar
        progress.update(1)

    # close the tqdm progress bar
    progress.close()

    print("\nDataset was successfully labeled!")

# initiator
if __name__ == "__main__":

    os.system("cls")
    # raw dataset filename
    unlabeledDatasetName = input("\nEnter the unlabelled dataset's name you want to use in 01_dataset folder (.pickle): ")
    unlabeledDatasetName = "./01_dataset/" + unlabeledDatasetName
    # labelled dataset filename
    labeledDatasetName = input("Enter the labelled dataset's name you want to save in 01_dataset folder (.csv): ")
    labeledDatasetName = "01_dataset/" + labeledDatasetName
    # randomization key to make the dataset reproducible
    seedKey = int(input("Enter the randomization key: "))

    # perform the dataset priming operation and save it as CSV file
    datasetWriter(
        seedKey,
        unlabeledDatasetName,
        None,
        labeledDatasetName
    )
