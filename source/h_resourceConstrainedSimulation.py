#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

# share variables that has the similar values
import a_sharedVariableSetting as sVS
# reuse the codes to create the unlabeled dataset
import b_unlabeledDatasetGenerator as uDG
# reuse the code to create the labeled dataset
import c_trainingDatasetLabeler as tDL
# reuse the codes to use the task scheduling switcher component
from f_taskSchedulingSwitcherBenchmark import fallbackAlgorithm
from f_taskSchedulingSwitcherBenchmark import schedulingAlgorithmSwitcherH5
from f_taskSchedulingSwitcherBenchmark import schedulingAlgorithmSwitcherTFLite

# module for
# 1. running the processes in a multi-cpu setup
# 2. creating a shared memory to share the variables
# 3. sending the output from process to main
from multiprocessing import Process, Value, Queue
# specify the ctypes used in the shared memory
from ctypes import c_bool, c_int, c_float

# random number generation module
from random import uniform, seed, randint, shuffle
# for recording the computation time and loop waiting
from time import perf_counter, sleep
# to color warning text with red color
from colorama import Fore, Style
# load uber-cool loading bar
from tqdm import tqdm
# for finding specific files
from glob import glob
# reading data from pickle
import pickle
# get os specific feature
import os

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
            typeOfDevices["securityDoor0"][1]
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

""" create the list of tasks to be scheduled in the span of one hour for one day """

# function to generate the task to be scheduled in span of 1 hour
def hourlyTasksToBeScheduled(hourCycle, typeOfDevices, numberOfPeople):

    hourlyTasks = {}

    for hour in numberOfPeople.keys():

        tempListOfTasksInOneHour = {}

        # for each cycle of tasks to be scheduled in one hour
        for cycle in range(hourCycle):

            # get the list of activated device unit based on the number of people at that time
            tempListOfTasksInOneHour[cycle] = deviceUnitMultiplier(typeOfDevices, numberOfPeople[hour])

        hourlyTasks[hour] = tempListOfTasksInOneHour

    return hourlyTasks

""" create a scheduling reference to compare it with the scheduler created by the framework """

# function to automate the scheduling of the tasks via FCFS, SJF, SnF, OnF
def createSchedulingReference(resourceConstraints, hourlyTasks):

    schedulingReference = {}
    print("\nGenerating the scheduling reference for comparison with the Task Scheduling Switcher Framework.")
    # define the tqdm progress bar
    progress = tqdm(total=len(hourlyTasks))

    for hour in hourlyTasks.keys():

        tempHourResult = {}

        for cycle in hourlyTasks[hour]:

            tempCycleResult = {}

            # start the timer
            timer0 = perf_counter()
            # compute the scheduler by using FCFS
            _, schedulerDuration = uDG.firstComeFirstServe(resourceConstraints, hourlyTasks[hour][cycle])
            # stop the timer
            timer0 = perf_counter() - timer0

            tempCycleResult["firstComeFirstServe"] = {"computationTime": timer0, "schedulerDuration": schedulerDuration}

            # start the timer
            timer1 = perf_counter()
            # compute the scheduler by using SJF (actually shortest job next can be called as shortest job first)
            _, schedulerDuration = uDG.shortestJobNext(resourceConstraints, hourlyTasks[hour][cycle])
            # stop the timer
            timer1 = perf_counter() - timer1

            tempCycleResult["shortestJobNext"] = {"computationTime": timer1, "schedulerDuration": schedulerDuration}

            # start the timer
            timer2 = perf_counter()
            # compute the scheduler by using SnF
            _, schedulerDuration = uDG.sortAndFit(resourceConstraints, hourlyTasks[hour][cycle])
            # stop the timer
            timer2 = perf_counter() - timer2

            tempCycleResult["sortAndFit"] = {"computationTime": timer2, "schedulerDuration": schedulerDuration}

            # start the timer
            timer3 = perf_counter()
            # compute the scheduler by using OnF
            _, schedulerDuration = uDG.octasortAndFit(resourceConstraints, hourlyTasks[hour][cycle])
            # stop the timer
            timer3 = perf_counter() - timer3

            tempCycleResult["octasortAndFit"] = {"computationTime": timer3, "schedulerDuration": schedulerDuration}

            tempHourResult[cycle] = tempCycleResult

        schedulingReference[hour] = tempHourResult
        # update the tqdm progress bar
        progress.update(1)

    # close the tqdm progress bar
    progress.close()

    return schedulingReference

""" analyze the minimum and maximum computation time from the unlabeled dataset """

# function to analyze the minimum and maximum computation value recorded in the unlabeled dataset
def analyzeUnlabeledDatasetForMinMaxComputationValue():

    # list all unlabeled dataset .pickle file exist in 01_dataset/ folder
    pickleFound = [
        file
        for path, subdir, files in os.walk("./01_dataset/")
        for file in glob(os.path.join(path, "*.pickle"))
    ]

    bestTime = 0
    worstTime = 0

    for i in range(len(pickleFound)):

        # read and load the pickle file to the memory
        unlabeledDataset = pickle.load(open(pickleFound[i], "rb"))
        # temp for storing the min/max value
        minVal = float("inf")
        maxVal = float("-inf")

        # finding min/max value
        for j in range(len(unlabeledDataset)):

            for schedulingAlgorithm in unlabeledDataset[j]["schedulingMethod"].keys():

                val = unlabeledDataset[j]["schedulingMethod"][schedulingAlgorithm]["computationTime"]

                if (val < minVal):

                    minVal = val

                if (val > maxVal):

                    maxVal = val

        bestTime += minVal
        worstTime += maxVal

    # get the average min max value
    bestTime /= len(pickleFound)
    worstTime /= len(pickleFound)

    return bestTime, worstTime

""" calculate the computation deadline """

# function to set the scheduler computation deadline
def deadlineValue(worstTime, bestTime, people):

    d = bestTime + worstTime / people

    return d

# initiator
if __name__ == "__main__":

    os.system("cls")
    # because it will affect the dataset creation time measurement
    print(Fore.RED + "\n1. Make sure that you are launching this simulation program without any background tasks running!")
    print(Fore.RED + "2. The simulation program must run on the same computer where the training datasets were created!" + Style.RESET_ALL)
    # NOTE: use .h5 trained file for a standard tensorflow model, use .tflite trained file for a quantize model (an optimized
    #       model with low latency processing although it has a small trade-off in accuracy)
    AImodelName = "./04_simulate/" + input("\nEnter the AI model to be use from the 04_simulate folder (.h5 or .tflite file format): ")
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

    # get the average best/worst computation time recorded in unlabeled dataset
    bestTime, worstTime = analyzeUnlabeledDatasetForMinMaxComputationValue()

    print("\nBest computation time: " + str(bestTime) + "s")
    print("Worst computation time: " + str(worstTime) + "s")

    # generate the task to be scheduled hourly where in one hour, the scheduling framework will schedule the task for sVS.hourCycle times
    hourlyTasks = hourlyTasksToBeScheduled(sVS.hourCycle, typeOfDevices, numberOfPeople)
    # create a scheduling reference to compare the our proposed framework with normal scheduling method
    schedulingReference = createSchedulingReference(resourceConstraints, hourlyTasks)

    mode = None

    # user inputed a standard model
    if (".h5" in AImodelName):

        mode = True

    # user inputed a quantize model
    elif (".tflite" in AImodelName):

        mode = False

    # one latch to kill the processes via shared memory
    killProcess = Value(c_bool, False)
    # initialize a queue (put)
    schedulingInputSwitch = Queue()
    schedulingInputFallback = Queue()

    """ preparing the scheduling algorithm switcher """

    # initialize a latch to control the schedulingAlgorithmSwitcher process via shared memory
    readyOrNotSwitcher = Value(c_bool, False)
    readyOrNotFallback = Value(c_bool, False)
    schedulingSwitcherState = Value(c_bool, False)
    # initialize a latch to manage the schedulingAlgorithmSwitcher model input via shared memory
    numberOfResourceConstraints = Value(c_float, 0.0)
    numberOfTasks = Value(c_float, 0.0)
    computationDeadline = Value(c_float, 0.0)

    # initialize a queue (get)
    chosenAlgorithm = Queue()
    switcherScheduler = Queue()
    switcherComputationTime = Queue()

    # using the .h5 model
    if (mode is True):

        # define the schedulingAlgorithmSwitcher process
        p1 = Process(
            target=schedulingAlgorithmSwitcherH5,
            args=(
                AImodelName,
                killProcess,
                readyOrNotSwitcher,
                schedulingSwitcherState,
                numberOfResourceConstraints,
                numberOfTasks,
                computationDeadline,
                chosenAlgorithm,
                schedulingInputSwitch,
                switcherScheduler,
                switcherComputationTime
            )
        )

    # using the .tflite model
    else:

        # define the schedulingAlgorithmSwitcher process
        p1 = Process(
            target=schedulingAlgorithmSwitcherTFLite,
            args=(
                AImodelName,
                killProcess,
                readyOrNotSwitcher,
                schedulingSwitcherState,
                numberOfResourceConstraints,
                numberOfTasks,
                computationDeadline,
                chosenAlgorithm,
                schedulingInputSwitch,
                switcherScheduler,
                switcherComputationTime
            )
        )

    # start the processes
    p1.start()

    """ preparing a fallback scheduling algorithm """

    # initialize a latch to control the fallbackAlgorithm process via shared memory
    fallbackAlgorithmState = Value(c_bool, False)

    # initialize a queue (get)
    fallbackScheduler = Queue()
    fallbackComputationTime = Queue()

    # define the fallbackAlgorithm process
    p2 = Process(
        target=fallbackAlgorithm,
        args=(
            killProcess,
            readyOrNotFallback,
            fallbackAlgorithmState,
            schedulingInputFallback,
            fallbackScheduler,
            fallbackComputationTime
        )
    )

    # start the processes
    p2.start()

    # wait until the ai model is fully loaded
    while ((readyOrNotSwitcher is False) and (readyOrNotFallback is False)):

        sleep(0.1)

    """ start the simulation """

    print("\nRunning the scheduling simulation.")
    # define the tqdm progress bar
    progress = tqdm(total=len(numberOfPeople))

    # storing the simulation result
    simulationResult = {}
    # storing the average deadline accuracy for the scheduling framework
    sFdeadlineAccuracy = 0
    # storing the average deadline accuracy for the scheduling reference
    sRdeadlineAccuracy = 0
    # storing the average classification accuracy of the MTL model
    classificationAccuracy = 0

    process1Succeed = 0
    process2Succeed = 0
    bothProcessesFailed = 0

    # storing the average optimization
    onfOptimizationDegree = 0
    snfOptimizationDegree = 0
    sjfOptimizationDegree = 0
    fcfsOptimizationDegree = 0
    referenceOptimizationDegree = 0
    frameworkOptimizationDegree = 0

    # deadline accuracy
    onfDeadlineAccuracy = 0
    snfDeadlineAccuracy = 0
    sjfDeadlineAccuracy = 0
    fcfsDeadlineAccuracy = 0

    for hour in numberOfPeople.keys():

        hourlyResult = {}
        # computation deadline for that hour
        d = deadlineValue(worstTime, bestTime, numberOfPeople[hour])

        for i in range(sVS.hourCycle):

            cycleResult = {}

            """ feed the scheduling parameter into the scheduling framework """

            # normalize (min-max) the ai model input for the number of resource constraint and update the value on the shared memory
            numberOfResourceConstraints.value = (
                len(resourceConstraints) - sVS.numberOfResourceConstraints_range[0]
            ) / (
                sVS.numberOfResourceConstraints_range[1] - sVS.numberOfResourceConstraints_range[0]
            )
            # normalize (min-max) the ai model input for the number of tasks and update the value on the shared memory
            numberOfTasks.value = (
                len(hourlyTasks[hour][i]) - sVS.numberOfTasks_range[0]
            ) / (
                sVS.numberOfTasks_range[1] - sVS.numberOfTasks_range[0]
            )
            # # normalize (min-max) the ai model input for the computation deadline and update the value on the shared memory
            computationDeadline.value = (d - (sVS.latencyH5 if mode is True else sVS.latencyTflite) - 0) / (sVS.maxComputationDeadline - 0)

            # load the task to be scheduled and resource constraints into the input
            # NOTE: reasoning for putting the same value? data passed via queue is usable one time only
            schedulingInputSwitch.put(
                {
                    "resourceConstraints": resourceConstraints,
                    "taskToBeScheduled": hourlyTasks[hour][i]
                }
            )
            schedulingInputFallback.put(
                {
                    "resourceConstraints": resourceConstraints,
                    "taskToBeScheduled": hourlyTasks[hour][i]
                }
            )

            """ start the scheduling """

            # change the processes state from waiting to scheduling
            schedulingSwitcherState.value = True
            fallbackAlgorithmState.value = True

            """ get the scheduling result """

            # get the queue from the switcher process
            chosenAlgorithmOut = chosenAlgorithm.get()
            switcherSchedulerOut = switcherScheduler.get()
            switcherComputationTimeOut = switcherComputationTime.get()
            # get the queue from the fallback process
            fallbackSchedulerOut = fallbackScheduler.get()
            fallbackComputationTimeOut = fallbackComputationTime.get()

            """ check if the scheduling framework choose the correct scheduling algorithm """

            # computation time from the scheduling reference
            computationTimeReference = [
                schedulingReference[hour][i]["octasortAndFit"]["computationTime"],
                schedulingReference[hour][i]["sortAndFit"]["computationTime"],
                schedulingReference[hour][i]["shortestJobNext"]["computationTime"],
                schedulingReference[hour][i]["firstComeFirstServe"]["computationTime"]
            ]
            schedulerDurationReference = [
                schedulingReference[hour][i]["octasortAndFit"]["schedulerDuration"],
                schedulingReference[hour][i]["sortAndFit"]["schedulerDuration"],
                schedulingReference[hour][i]["shortestJobNext"]["schedulerDuration"],
                schedulingReference[hour][i]["firstComeFirstServe"]["schedulerDuration"]
            ]

            # find the algorithm computation time that completed under the computation deadline
            validComputationTime = [time for time in computationTimeReference if time < d]

            # if there is any valid algorithm computation time
            if (validComputationTime != []):

                sRdeadlineAccuracy += 1
                # find the computation time closest with the computation deadline
                nearestComputationTime = max(validComputationTime)

            else:

                # if there is no valid computation time, use the smallest one
                nearestComputationTime = min(computationTimeReference)

            # find the closest computation time position on the list
            # NOTE: this position represent the correct scheduling algorithm where
            #       0 = OnF, 1 = SnF, 2 = SJF, 3 = FCFS
            correctSchedulingAlgorithm = computationTimeReference.index(nearestComputationTime)

            # check if the model choose the correct scheduling algorithm
            if (correctSchedulingAlgorithm == chosenAlgorithmOut):

                classificationAccuracy += 1

            """ check if the scheduling framework manage to meet the computation deadline """

            meetTheComputationDeadline = False
            # True = process 1 scheduler; False = process 2 scheduler
            schedulerType = None

            # check if the switcher computation time is under the deadline
            if (d > switcherComputationTimeOut):

                # scheduler from the process 1 is used
                meetTheComputationDeadline = True
                schedulerType = True
                sFdeadlineAccuracy += 1
                process1Succeed += 1

            # switcher failed to compute under the deadline; use fallback solution
            else:

                # switcher failed to compute under the deadline; use fallback solution
                if (d > fallbackComputationTimeOut):

                    # scheduler from the process 2 is used
                    meetTheComputationDeadline = True
                    schedulerType = False
                    sFdeadlineAccuracy += 1
                    process2Succeed += 1

                # both failed
                else:

                    bothProcessesFailed += 1

                    if (switcherComputationTimeOut < fallbackComputationTimeOut):

                        # scheduler from the process 1 is used
                        schedulerType = True

                    else:

                        # scheduler from the process 2 is used
                        schedulerType = False

            """ add the scheduler execution duration to be calculated later for finding the optimization degree """

            onfOptimizationDegree += schedulingReference[hour][i]["octasortAndFit"]["schedulerDuration"]
            snfOptimizationDegree += schedulingReference[hour][i]["sortAndFit"]["schedulerDuration"]
            sjfOptimizationDegree += schedulingReference[hour][i]["shortestJobNext"]["schedulerDuration"]
            fcfsOptimizationDegree += schedulingReference[hour][i]["firstComeFirstServe"]["schedulerDuration"]
            referenceOptimizationDegree += schedulerDurationReference[correctSchedulingAlgorithm]
            frameworkOptimizationDegree += switcherSchedulerOut["duration"] if (schedulerType is True) else fallbackSchedulerOut["duration"]

            onfDeadlineAccuracy += 1 if (schedulingReference[hour][i]["octasortAndFit"]["computationTime"] < fallbackComputationTimeOut) else 0
            snfDeadlineAccuracy += 1 if (schedulingReference[hour][i]["sortAndFit"]["computationTime"] < fallbackComputationTimeOut) else 0
            sjfDeadlineAccuracy += 1 if (schedulingReference[hour][i]["shortestJobNext"]["computationTime"] < fallbackComputationTimeOut) else 0
            fcfsDeadlineAccuracy += 1 if (schedulingReference[hour][i]["firstComeFirstServe"]["computationTime"] < fallbackComputationTimeOut) else 0

            cycleResult = {
                "correctSchedulingAlgorithm":
                    "octasortAndFit" if (correctSchedulingAlgorithm == 0) else
                    "sortAndFit" if (correctSchedulingAlgorithm == 1) else
                    "shortestJobFirst" if (correctSchedulingAlgorithm == 2) else
                    "firstComeFirstServe",
                "chosenSchedulingAlgorithm":
                    "octasortAndFit" if (chosenAlgorithmOut == 0) else
                    "sortAndFit" if (chosenAlgorithmOut == 1) else
                    "shortestJobFirst" if (chosenAlgorithmOut == 2) else
                    "firstComeFirstServe",
                "computationDeadline":
                    d,
                "referenceSchedulerComputationTime":
                    nearestComputationTime,
                "schedulingFrameworkComputationTime":
                    switcherComputationTimeOut if (d > switcherComputationTimeOut) else
                    fallbackComputationTimeOut if (d > fallbackComputationTimeOut) else
                    switcherComputationTimeOut if (switcherComputationTimeOut < fallbackComputationTimeOut) else
                    fallbackComputationTimeOut,
                "framewokMeetTheComputationDeadline":
                    meetTheComputationDeadline,
                "frameworkSchedulerUsed":
                    "taskSchedulingSwitcher" if (schedulerType is True) else
                    "fallbackSchedulingAlgorithm",
                "frameworkSchedulerDuration":
                    switcherSchedulerOut["duration"] if (schedulerType is True) else
                    fallbackSchedulerOut["duration"],
                "referenceSchedulerDuration":
                    schedulerDurationReference[correctSchedulingAlgorithm]
            }

            hourlyResult[i] = cycleResult

            # just in case, wait until the scheduling processes are not in running state before we reloop the simulation
            while ((schedulingSwitcherState is True) and (fallbackAlgorithmState is True)):

                sleep(0.1)

        # update the tqdm progress bar
        progress.update(1)

        simulationResult[hour] = hourlyResult

    # close the tqdm progress bar
    progress.close()

    killProcess.value = True
    p1.join()
    p2.join()

    """ refining the result """

    numberofSchedulerProduced = len(numberOfPeople) * sVS.hourCycle
    # get the average computation deadline accuracy
    sFdeadlineAccuracy = sFdeadlineAccuracy / numberofSchedulerProduced * 100
    sRdeadlineAccuracy = sRdeadlineAccuracy / numberofSchedulerProduced * 100
    onfDeadlineAccuracy = onfDeadlineAccuracy / numberofSchedulerProduced * 100
    snfDeadlineAccuracy = snfDeadlineAccuracy / numberofSchedulerProduced * 100
    sjfDeadlineAccuracy = sjfDeadlineAccuracy / numberofSchedulerProduced * 100
    fcfsDeadlineAccuracy = fcfsDeadlineAccuracy / numberofSchedulerProduced * 100
    # get the average classification accuracy
    classificationAccuracy = classificationAccuracy / numberofSchedulerProduced * 100
    # get the average scheduler optimization degree
    frameworkOptimizationDegree = 100 - frameworkOptimizationDegree / fcfsOptimizationDegree * 100
    referenceOptimizationDegree = 100 - referenceOptimizationDegree / fcfsOptimizationDegree * 100
    onfOptimizationDegree = 100 - onfOptimizationDegree / fcfsOptimizationDegree * 100
    snfOptimizationDegree = 100 - snfOptimizationDegree / fcfsOptimizationDegree * 100
    sjfOptimizationDegree = 100 - sjfOptimizationDegree / fcfsOptimizationDegree * 100
    fcfsOptimizationDegree = 100 - fcfsOptimizationDegree / fcfsOptimizationDegree * 100
    # get the average of process utilization
    process1Succeed = process1Succeed / numberofSchedulerProduced * 100
    process2Succeed = process2Succeed / numberofSchedulerProduced * 100
    bothProcessesFailed = bothProcessesFailed / numberofSchedulerProduced * 100

    """ show the results """

    print("\nAverage result analyzed from " + str(sVS.hourCycle * len(numberOfPeople)) + " schedulers.")
    print("  =Scheduling reference optimization degree: " + str(referenceOptimizationDegree) + "%")
    print("  =Scheduling framework optimization degree: " + str(frameworkOptimizationDegree) + "%")
    print("  =Scheduling reference deadline accuracy: " + str(sRdeadlineAccuracy) + "%")
    print("  =Scheduling framework deadline accuracy: " + str(sFdeadlineAccuracy) + "%")
    print("  =Sortest Job First optimization accuracy: " + str(sjfOptimizationDegree) + "%")
    print("  =Sortest Job First deadline accuracy: " + str(sjfDeadlineAccuracy) + "%")
    print("  =Classification accuracy: " + str(classificationAccuracy) + "%")
    print("  =Process 1 utilization: " + str(process1Succeed) + "%")
    print("  =Process 2 utilization: " + str(process2Succeed) + "%")
    print("  =Both processes failed: " + str(bothProcessesFailed) + "%")

    print("\nHourly result")

    for hour in numberOfPeople.keys():

        print("  Hour " + str(hour))

        for i in range(sVS.hourSample):

            print("    Sample " + str(i))
            print("      =Correct scheduling algorithm: " + str(simulationResult[hour][i]["correctSchedulingAlgorithm"]))
            print("      =Chosen scheduling algorithm: " + str(simulationResult[hour][i]["chosenSchedulingAlgorithm"]))
            print("      =Computation deadline: " + str(simulationResult[hour][i]["computationDeadline"]) + "s")
            print("      =Scheduling reference computation time: " + str(simulationResult[hour][i]["referenceSchedulerComputationTime"]) + "s")
            print("      =Scheduling framework computation time: " + str(simulationResult[hour][i]["schedulingFrameworkComputationTime"]) + "s")
            print("      =Reference meet the computation deadline: " + str(
                True if simulationResult[hour][i]["computationDeadline"] > simulationResult[hour][i]["referenceSchedulerComputationTime"] else False)
            )
            print("      =Framework meet the computation deadline: " + str(simulationResult[hour][i]["framewokMeetTheComputationDeadline"]))
            print("      =Scheduler used by the framework: " + str(simulationResult[hour][i]["frameworkSchedulerUsed"]))
            print("      =Scheduling framework scheduler duration: " + str(simulationResult[hour][i]["frameworkSchedulerDuration"]) + "s")
            print("      =Scheduling reference scheduler duration: " + str(simulationResult[hour][i]["referenceSchedulerDuration"]) + "s")

        print("      ...")
