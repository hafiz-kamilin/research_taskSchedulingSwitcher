#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

# module for
# 1. running the processes in a multi-cpu setup
# 2. creating a shared memory to share the variables
# 3. sending the output from process to main
from multiprocessing import Process, Value, Queue
# specify the ctypes used in the shared memory
from ctypes import c_bool, c_int, c_float

# share variables that has the similar values
import a_sharedVariableSetting as sVS
# reuse the codes to create the unlabeled dataset
import b_unlabeledDatasetGenerator as uDG
# reuse the codes to label the unlabeled dataset
import c_trainingDatasetLabeler as tDL

# for recording the computation time and loop waiting
from time import perf_counter, sleep
# to color warning text with red color
from colorama import Fore, Style
# load uber-cool loading bar
from tqdm import tqdm
# get os specific feature
import os

""" a process that uses deep learning to switch the scheduling algorithm and schedule the tasks on the fly """

# process to switch the scheduling algorithm based on the parameters given and perform the scheduling (using .h5 model)
def schedulingAlgorithmSwitcherH5(
    # ai model to be used (str)
    AImodelName,
    # latch control (shared memory)
    killProcess,
    readyOrNotSwitcher,
    schedulingSwitcherState,
    # ai model input (shared memory)
    numberOfResourceConstraints,
    numberOfTasks,
    computationDeadline,
    # ai model output (queue put)
    choosenAlgorithm,
    # tasks to be scheduled (queue get)
    schedulingInputSwitch,
    # generated scheduler (queue put)
    switcherScheduler,
    # computation time (queue put)
    switcherComputationTime
):

    # supress tensorflow debug message
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # load tensorflow as machine learning framework
    import tensorflow as tf
    # for data processing
    import numpy as np

    # define the dict that store the scheduling algorithms available for choosing
    schedulerAlgorithms = {
        0: uDG.octasortAndFit,
        1: uDG.sortAndFit,
        2: uDG.shortestJobNext,
        3: uDG.firstComeFirstServe
    }
    # load the model
    model = tf.keras.models.load_model(AImodelName)

    # tell the main that the model was loaded and this proces is in waiting state
    readyOrNotSwitcher.value = True

    # as long as the killProcess is not activated, repeat the tasks
    while (killProcess.value is False):

        # if there are tasks that need to be scheduled
        if (schedulingSwitcherState.value is True):

            # get the scheduling input from the queue
            sI = schedulingInputSwitch.get()

            # start the timer
            timer = perf_counter()

            # convert the values into something that tensorflow can process
            x = tf.Variable(
                [
                    np.array(
                        [
                            numberOfResourceConstraints.value,
                            numberOfTasks.value,
                            computationDeadline.value
                        ]
                    )
                ],
                trainable=True,
                dtype=tf.float64
            )

            # pass the parameters to the model and get the answer
            choosen = model(x)
            # for storing the result
            schedulerResult = {}
            # find the number of scheduling algorithm with the highest confidence
            choosen = [i for i, j in enumerate(choosen) if j == max(choosen)][0]
            # schedule the input based on the algorithm with the highest confidence
            schedulerResult["scheduler"], schedulerResult["duration"] = schedulerAlgorithms[choosen](
                sI["resourceConstraints"],
                sI["taskToBeScheduled"]
            )

            # stop the timer
            timer = perf_counter() - timer

            # send the results using queue
            choosenAlgorithm.put(choosen)
            switcherScheduler.put(schedulerResult)
            switcherComputationTime.put(timer)
            # always reset the if state checking back to original value
            schedulingSwitcherState.value = False

# process to switch the scheduling algorithm based on the parameters given and perform the scheduling (using .tflite model)
def schedulingAlgorithmSwitcherTFLite(
    # ai model to be used (str)
    AImodelName,
    # latch control (shared memory)
    killProcess,
    readyOrNotSwitcher,
    schedulingSwitcherState,
    # ai model input (shared memory)
    numberOfResourceConstraints,
    numberOfTasks,
    computationDeadline,
    # ai model output (queue put)
    choosenAlgorithm,
    # tasks to be scheduled (queue get)
    schedulingInputSwitch,
    # generated scheduler (queue put)
    switcherScheduler,
    # computation time (queue put)
    switcherComputationTime
):

    # supress tensorflow debug message
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # load tensorflow as machine learning framework
    import tensorflow as tf
    # for data processing
    import numpy as np

    # define the dict that store the scheduling algorithms available for choosing
    schedulerAlgorithms = {
        0: uDG.octasortAndFit,
        1: uDG.sortAndFit,
        2: uDG.shortestJobNext,
        3: uDG.firstComeFirstServe
    }
    # load tf lite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=AImodelName)
    interpreter.allocate_tensors()
    # get input and output tensors
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()
    # get the input shape of the model
    inputShape = inputDetails[0]["shape"]

    # tell the main that the model was loaded and this proces is in waiting state
    readyOrNotSwitcher.value = True

    # as long as the killProcess is not activated, repeat the tasks
    while (killProcess.value is False):

        # if there are tasks that need to be scheduled
        if (schedulingSwitcherState.value is True):

            # get the scheduling input from the queue
            sI = schedulingInputSwitch.get()

            # start the timer
            timer = perf_counter()

            # convert the values into something that tensorflow lite can process
            inputData = np.array(
                [
                    numberOfResourceConstraints.value,
                    numberOfTasks.value,
                    computationDeadline.value
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
            # for storing the result
            schedulerResult = {}
            # find the number of scheduling algorithm with the highest confidence
            choosen = [i for i, j in enumerate(choosen) if j == max(choosen)][0]
            # schedule the input based on the algorithm with the highest confidence
            schedulerResult["scheduler"], schedulerResult["duration"] = schedulerAlgorithms[choosen](
                sI["resourceConstraints"],
                sI["taskToBeScheduled"]
            )

            # stop the timer
            timer = perf_counter() - timer

            # send the results using queue
            choosenAlgorithm.put(choosen)
            switcherScheduler.put(schedulerResult)
            switcherComputationTime.put(timer)
            # always reset the if state checking back to original value
            schedulingSwitcherState.value = False

"""" process to perform preemptive algorithm choosen in case the the switcher choose a wrong scheduling algorithm
that take more time to compute. for this case, because shortestJobNext has the fastest computation time, we
use it as default backup scheduling algorithm. """

# modified "Shortest Job Next" for multiprocessing
def fallbackAlgorithm(
    # latch control (shared memory)
    killProcess,
    readyOrNotFallback,
    fallbackAlgorithmState,
    # tasks to be scheduled (queue get)
    schedulingInputFallback,
    # generated scheduler (queue put)
    fallbackScheduler,
    # computation time (queue put)
    fallbackComputationTime
):

    # function to check if the resource to be taken by the iot task is enough or not
    def constraintChecker(rC, rT, tTBT):

        canBeIncluded = True

        # for every resources labelled by the iot task to be taken
        for label in tTBT.keys():

            # if the resourse to be taken exceed the constraint
            if (rC[label] < rT[label] + tTBT[label]):

                canBeIncluded = False
                break

        return canBeIncluded

    # tell the main that this proces is in waiting state
    readyOrNotFallback.value = True

    # as long as the killProcess is not activated, repeat the tasks
    while (killProcess.value is False):

        # if there are tasks that need to be scheduled
        if (fallbackAlgorithmState.value is True):

            # get the scheduling input from the queue
            sI = schedulingInputFallback.get()

            # start the timer
            timer = perf_counter()

            # for storing the result
            schedulerResult = {}
            # sort according to the runtime length in ascending order
            sortedExecution = sorted(sI["taskToBeScheduled"], key=lambda x: x[1])
            # for storing the scheduler duration and result
            schedulerDuration = 0
            generatedScheduler = {}
            # dictionary to record every resource taken
            resourceTaken = {}

            # initialize the dictionary according to the number of constraints to be recorded
            for constraint in sI["resourceConstraints"].keys():

                resourceTaken[constraint] = 0

            # initialize the number of iot task to be assigned
            toBeAssignedTasks = len(sI["taskToBeScheduled"])
            # to track how many time the execution step has occured
            timeStep = 0
            # to store the shortest execution time
            shortestTime = 0
            # to store the fitting result temporarily
            temp = []
            # to record the iot tasks that are currently running in simulation right now
            runningTasks = []

            # perform the fitting task as long there are remaining tasks
            while (toBeAssignedTasks != 0):

                # due to the addition or subtraction in float format could never be a 0
                for constraint in sI["resourceConstraints"].keys():

                    # round off all the numbers in dictionary to a 3 decimal points
                    # having an accuracy in 3 decimal points is good enough
                    resourceTaken[constraint] = round(resourceTaken[constraint], 3)

                # as long the toBeAssignedTasks is not empty
                while (toBeAssignedTasks != 0):

                    # check the iot task if it can be fitted or not
                    if (constraintChecker(sI["resourceConstraints"], resourceTaken, sortedExecution[0][2]) is True):

                        # reduce the counter by 1 because 1 iot task is assigned
                        toBeAssignedTasks -= 1
                        # temporarily save the iot task
                        temp.append(sortedExecution[0])
                        # record the iot task as being executed right now (increment the iot task's runtime according to the timeStep)
                        runningTasks.append([sortedExecution[0][0], sortedExecution[0][1] + timeStep, sortedExecution[0][2]])

                        # record every resources consumed by the iot task
                        for constraint in sortedExecution[0][2].keys():

                            resourceTaken[constraint] += sortedExecution[0][2][constraint]

                        # delete the task from the list
                        del sortedExecution[0]

                    else:

                        # stop looking (for now)
                        break

                # if the fitter found iot tasks that can be executed
                if (temp != []):

                    # save the iot tasks with the time for it to run as dictionary key
                    generatedScheduler[timeStep] = temp
                    # reset the temp
                    temp = []

                # if there are iot tasks that are currently running
                if (runningTasks != []):

                    # find the iot task that will complete their task soon
                    # NOTE: the iot task that will complete their task soon has the shortest running time
                    shortestTime = min(runningTasks, key=lambda i: i[1])[1]
                    # record the time for the iot task that will complete their task soon
                    # NOTE: at the time for the iot task that will complete their task soon, iot system resources will be freed!
                    timeStep += abs(timeStep - shortestTime)

                    # for recording the tasks that recently completed it task
                    completedTasks = []

                    # for every iot tasks that are running right now
                    for task in range(len(runningTasks)):

                        # find the iot task that have the shortest runtime
                        if (runningTasks[task][1] == shortestTime):

                            # record the running iot task as completed
                            completedTasks.append(runningTasks[task])

                            # for every resource taken by the iot task
                            for resource in runningTasks[task][2].keys():

                                # freed the resource taken by the iot task
                                resourceTaken[resource] -= runningTasks[task][2][resource]

                    # remove the tasks that are marked for completion
                    runningTasks = [x for x in runningTasks if x not in completedTasks]

            # NOTE: this "if logic" work to find the scheduler duration
            # first take a look on iot tasks that are still running
            if (runningTasks != []):

                # iot tasks with the highest runtime in the runningTasks mark the scheduler duration
                schedulerDuration = max(runningTasks, key=lambda i: i[1])[1]

            # if the runningTasks is empty (all the tasks has been marked as completed)
            else:

                # the shortestTime mark the scheduler duration
                schedulerDuration = shortestTime

            # stop the timer
            timer = perf_counter() - timer
            # send the results using queue
            fallbackScheduler.put({"scheduler": generatedScheduler, "duration": schedulerDuration})
            fallbackComputationTime.put(timer)
            # always reset the if state checking back to original value
            fallbackAlgorithmState.value = False

""" create an unlabeled dataset and label it using a controllable computation deadline """

# function to create a labeled dataset
def createdDataset():

    # create a dataset that contain "sVS.cycleSwitch" of unlabeled data
    cycle = sVS.cycleSwitch
    # randomization key to make the dataset reproducible
    seedKey = int(input("Enter the randomization key: "))
    # probability for the resource constraint consumed by the device not to become zero
    flipCoinBias = sVS.flipCoinBias
    # create a small dataset to train the ai
    smallDataset = False
    # include the computation time taken for each scheduling algorithm
    withComputationTime = True
    # randomization range for the number of iot tasks in a system
    numberOfTasks_range = sVS.numberOfTasks_range
    # randomization range for the value of resource constraint in a system
    valueOfResourceConstraints_range = sVS.valueOfResourceConstraints_range
    # randomization range for the number of resource constraint in a system
    numberOfResourceConstraints_range = sVS.numberOfResourceConstraints_range

    # create an unlabeled dataset
    print("\nGenerating an unlabeled dataset.")
    unlabeledDataset = uDG.datasetRandomGenerator(
        cycle,
        seedKey,
        flipCoinBias,
        smallDataset,
        withComputationTime,
        numberOfTasks_range,
        valueOfResourceConstraints_range,
        numberOfResourceConstraints_range
    )

    # storing the generated label
    generatedLabel = {}

    print("\nLabeling up the unlabeled dataset.")
    # define the tqdm progress bar
    progress = tqdm(total=sVS.cycleSwitch)

    # create a label for the dataset
    for i in range(sVS.cycleSwitch):

        # label the dataset in a way each scheduling algorithm will have
        # an equal chance to meet the computation deadline
        generatedLabel[i] = tDL.datasetLabeler(unlabeledDataset, i)
        # update the tqdm progress bar
        progress.update(1)

    # close the tqdm progress bar
    progress.close()

    return unlabeledDataset, generatedLabel

# initiator
if __name__ == "__main__":

    os.system("cls")
    # because it will affect the dataset creation time measurement
    print(Fore.RED + "\n1. Make sure that you are launching this simulation program without any background tasks running!")
    print(Fore.RED + "2. The simulation program must run on the same computer where the training datasets were created!" + Style.RESET_ALL)
    # NOTE: use .h5 trained file for a standard tensorflow model, use .tflite trained file for a quantize model (an optimized
    #       model with low latency processing although it has a small trade-off in accuracy)
    AImodelName = "./02_model/" + input("\nEnter the AI model to be use from the 02_model folder (.h5 or .tflite file format): ")
    mode = None

    # user inputed a standard model
    if (".h5" in AImodelName):

        mode = True

    # user inputed a quantize model
    elif (".tflite" in AImodelName):

        mode = False

    # creating a labeled dataset
    unlabeledDataset, generatedLabel = createdDataset()
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
    choosenAlgorithm = Queue()
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
                choosenAlgorithm,
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
                choosenAlgorithm,
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

    # choosing the scheduling algorithm correctly
    aiModelAccuracy = 0
    # count how many time fallback scheduling algorithm is being used
    fallbackCallCount = 0
    # count how many time both methods failed to meet the computation deadline
    bothMethodsFailed = 0
    # preparing a dict to store the computation time accuracy between each scheduling algorithm
    computationDeadlineAccuracy = {
        "switcherFramework": 0,
        "octasortAndFit": 0,
        "sortAndFit": 0,
        "shortestJobNext": 0,
        "firstComeFirstServe": 0
    }
    # preparing a dict to store the optimization between each scheduling algorithm
    schedulerOptimization = {
        "switcherFramework": 0,
        "octasortAndFit": 0,
        "sortAndFit": 0,
        "shortestJobNext": 0,
        "firstComeFirstServe": 0
    }

    print("\nRunning the scheduling benchmark.")
    # define the tqdm progress bar
    progress = tqdm(total=sVS.cycleSwitch)

    # iterate for every unlabeled dataset that exist
    for i in range(sVS.cycleSwitch):

        # normalize (min-max) the ai model input for the number of resource constraint and update the value on the shared memory
        numberOfResourceConstraints.value = (
            len(unlabeledDataset[i]["resourceConstraints"]) - sVS.numberOfResourceConstraints_range[0]
        ) / (
            sVS.numberOfResourceConstraints_range[1] - sVS.numberOfResourceConstraints_range[0]
        )
        # normalize (min-max) the ai model input for the number of tasks and update the value on the shared memory
        numberOfTasks.value = (
            len(unlabeledDataset[i]["taskToBeScheduled"]) - sVS.numberOfTasks_range[0]
        ) / (
            sVS.numberOfTasks_range[1] - sVS.numberOfTasks_range[0]
        )

        # iterate for every label exist for that one unlabeled dataset
        for computationDeadlineKey in generatedLabel[i]:

            # load the task to be scheduled and resource constraints into the input
            # NOTE: reasoning for putting the same value? data passed via queue is usable one time only
            schedulingInputSwitch.put(
                {
                    "resourceConstraints": unlabeledDataset[i]["resourceConstraints"],
                    "taskToBeScheduled": unlabeledDataset[i]["taskToBeScheduled"]
                }
            )
            schedulingInputFallback.put(
                {
                    "resourceConstraints": unlabeledDataset[i]["resourceConstraints"],
                    "taskToBeScheduled": unlabeledDataset[i]["taskToBeScheduled"]
                }
            )

            # # normalize (min-max) the ai model input for the computation deadline and update the value on the shared memory
            computationDeadline.value = (computationDeadlineKey - (sVS.latencyH5 if mode is True else sVS.latencyTflite) - 0) / (sVS.maxComputationDeadline - 0)

            # change the processes state from waiting to scheduling
            schedulingSwitcherState.value = True
            fallbackAlgorithmState.value = True

            # get the queue from the switcher process
            choosenAlgorithmOut = choosenAlgorithm.get()
            switcherSchedulerOut = switcherScheduler.get()
            switcherComputationTimeOut = switcherComputationTime.get()
            # get the queue from the fallback process
            fallbackSchedulerOut = fallbackScheduler.get()
            fallbackComputationTimeOut = fallbackComputationTime.get()

            # get the correct answer for scheduling algorithm to be choosen
            correctAlgorithm = generatedLabel[i][computationDeadlineKey]
            correctAlgorithm = [i for i, j in enumerate(correctAlgorithm) if j == max(correctAlgorithm)][0]

            # check if the model choose the correct scheduling algorithm
            if (correctAlgorithm == choosenAlgorithmOut):

                aiModelAccuracy += 1

            # check if the switcher computation time is under the deadline
            if (computationDeadlineKey > switcherComputationTimeOut):

                computationDeadlineAccuracy["switcherFramework"] += 1
                schedulerOptimization["switcherFramework"] += switcherSchedulerOut["duration"]

            # switcher failed to compute under the deadline
            else:

                # switcher failed to compute under the deadline; use fallback solution
                if (computationDeadlineKey > fallbackComputationTimeOut):

                    schedulerOptimization["switcherFramework"] += fallbackSchedulerOut["duration"]
                    computationDeadlineAccuracy["switcherFramework"] += 1
                    fallbackCallCount += 1

                # both failed
                else:

                    schedulerOptimization["switcherFramework"] += switcherSchedulerOut["duration"] if \
                        (fallbackComputationTimeOut > switcherComputationTimeOut) else fallbackSchedulerOut["duration"]
                    bothMethodsFailed += 1

            # get the rest with typical scheduling algorithm in computation deadline accuracy
            if (unlabeledDataset[i]["schedulingMethod"]["octasortAndFit"]["computationTime"] < computationDeadlineKey):
                computationDeadlineAccuracy["octasortAndFit"] += 1
            if (unlabeledDataset[i]["schedulingMethod"]["sortAndFit"]["computationTime"] < computationDeadlineKey):
                computationDeadlineAccuracy["sortAndFit"] += 1
            if (unlabeledDataset[i]["schedulingMethod"]["shortestJobNext"]["computationTime"] < computationDeadlineKey):
                computationDeadlineAccuracy["shortestJobNext"] += 1
            if (unlabeledDataset[i]["schedulingMethod"]["firstComeFirstServe"]["computationTime"] < computationDeadlineKey):
                computationDeadlineAccuracy["firstComeFirstServe"] += 1

            # get the rest of typical scheduling algorithm in scheduler duration
            schedulerOptimization["octasortAndFit"] += unlabeledDataset[i]["schedulingMethod"]["octasortAndFit"]["schedulerDuration"]
            schedulerOptimization["sortAndFit"] += unlabeledDataset[i]["schedulingMethod"]["sortAndFit"]["schedulerDuration"]
            schedulerOptimization["shortestJobNext"] += unlabeledDataset[i]["schedulingMethod"]["shortestJobNext"]["schedulerDuration"]
            schedulerOptimization["firstComeFirstServe"] += unlabeledDataset[i]["schedulingMethod"]["firstComeFirstServe"]["schedulerDuration"]

            # just in case, wait until the scheduling processes are not in running state before we reloop the simulation
            while ((schedulingSwitcherState is True) and (fallbackAlgorithmState is True)):

                sleep(0.1)

        # update the tqdm progress bar
        progress.update(1)

    # close the tqdm progress bar
    progress.close()

    killProcess.value = True
    p1.join()
    p2.join()

    print(
        "\nTask Scheduling Switcher being used instead of Fallback Scheduling Algorithm: " + str(
            100 - (fallbackCallCount / (2500 * 4) * 100) - (bothMethodsFailed / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100)
        ) + "%"
    )
    print("Fallback Scheduling Algorithm being used instead of Task Scheduling Switcher: " + str(fallbackCallCount / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")
    print("Both method failed to meet the computation deadline: " + str(bothMethodsFailed / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")

    print("\nPrediction accuracy = " + str(aiModelAccuracy / (sVS.cycleSwitch * 4) * 100) + "%")

    print("\nDeadline accuracy")
    print("  =schedulingFramework: " + str(computationDeadlineAccuracy["switcherFramework"] / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")
    print("  =octasortAndFit: " + str(computationDeadlineAccuracy["octasortAndFit"] / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")
    print("  =sortAndFit: " + str(computationDeadlineAccuracy["sortAndFit"] / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")
    print("  =shortestJobNext: " + str(computationDeadlineAccuracy["shortestJobNext"] / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")
    print("  =firstComeFirstServe: " + str(computationDeadlineAccuracy["firstComeFirstServe"] / (sVS.cycleSwitch * sVS.numberOfAlgorithm) * 100) + "%")

    print("\nScheduler optimization (how much the scheduler duration is reduced) when compared with firstComeFirstServe")
    print("  =schedulingFramework: " + str(100 - schedulerOptimization["switcherFramework"] / schedulerOptimization["firstComeFirstServe"] * 100) + "%")
    print("  =octasortAndFit: " + str(100 - schedulerOptimization["octasortAndFit"] / schedulerOptimization["firstComeFirstServe"] * 100) + "%")
    print("  =sortAndFit: " + str(100 - schedulerOptimization["sortAndFit"] / schedulerOptimization["firstComeFirstServe"] * 100) + "%")
    print("  =shortestJobNext: " + str(100 - schedulerOptimization["shortestJobNext"] / schedulerOptimization["firstComeFirstServe"] * 100) + "%")
