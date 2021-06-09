#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

# random number generation module
from random import uniform, seed, randint
# to color warning text with red color
from colorama import Fore, Style
# stopwatch function using nanosecond
from time import perf_counter
# duplicate variable into a new memory
from copy import deepcopy
# load uber-cool loading bar
from tqdm import tqdm
# saving data into pickle
import pickle
# for os specific feature
import os

# share variables that has the similar values
import a_sharedVariableSetting as sVS

""" randomize the iot tasks to be scheduled and create a raw dataset to be primed """

# function to randomize the parameter generation in certain range for every cycle specified
def datasetRandomGenerator(
    cycle,
    seedKey,
    flipCoinBias,
    smallDataset,
    withComputationTime,
    numberOfTasks_range,
    valueOfResourceConstraints_range,
    numberOfResourceConstraints_range
):

    # make the randomization reproducable
    seed(seedKey)

    # for saving the randomly generated iot tasks
    dataset = {}

    # define the tqdm progress bar
    progress = tqdm(total=cycle)

    """ randomize the iot tasks runtime and constraints """

    for number in range(cycle):

        taskToBeScheduled = []

        # randomize (int) number of the iot tasks in a system
        numberOfTasks = randint(numberOfTasks_range[0], numberOfTasks_range[1])
        # randomize (int) number of the resource constraints in a system
        numberOfResourceConstraints = randint(numberOfResourceConstraints_range[0], numberOfResourceConstraints_range[1])
        # (re)initialize the dict
        resourceConstraints = {}

        # randomize (float) the resource constraint values
        for i in range(numberOfResourceConstraints):

            resourceConstraints[i] = round(uniform(valueOfResourceConstraints_range[0], valueOfResourceConstraints_range[1]), 3)

        # for every iot tasks that need to be generated
        for task in range(numberOfTasks):

            # for saving the newly generated resource to be taken by the iot task
            resourceToBeTaken = {}

            # randomize the iot task runtime based on the mean value of the resource constraint
            runtime = round(uniform(1, sum(resourceConstraints.values()) / len(resourceConstraints)), 3)

            # for every labelled resource constraint in the iot system
            for label in resourceConstraints.keys():

                # flip the coin with bias to get head; if it is head, randomize the resource constraint
                if (randint(1, 100) >= flipCoinBias):

                    # do note that we need to restrict the float to the 3 decimal points
                    resourceToBeTaken[label] = round(uniform(0.001, resourceConstraints[label]), 3)

                # if it is tail, the resource to be taken by the tasks is 0
                else:

                    resourceToBeTaken[label] = 0.000

            # append it to the dataset as tuple(task id, runtime, load)
            taskToBeScheduled.append(["task_" + str(task), runtime, resourceToBeTaken])

        if (withComputationTime is True):

            """ compute the scheduler using the octasort and fit """

            # start the stopwatch
            octasortAndFitComputationTime = perf_counter()
            _, octasortAndFitSchedulerDuration = octasortAndFit(resourceConstraints, taskToBeScheduled)
            # stop the stopwatch
            octasortAndFitComputationTime = perf_counter() - octasortAndFitComputationTime

            """ compute the scheduler using the sort and fit """

            # start the stopwatch
            sortAndFitComputationTime = perf_counter()
            _, sortAndFitSchedulerDuration = sortAndFit(resourceConstraints, taskToBeScheduled)
            # stop the stopwatch
            sortAndFitComputationTime = perf_counter() - sortAndFitComputationTime

            """ compute the scheduler using the shortest job next """

            # start the stopwatch
            shortestJobNextComputationTime = perf_counter()
            _, shortestJobNextSchedulerDuration = shortestJobNext(resourceConstraints, taskToBeScheduled)
            # stop the stopwatch
            shortestJobNextComputationTime = perf_counter() - shortestJobNextComputationTime

            """ compute the scheduler using the first come first serve """

            # start the stopwatch
            firstComeFirstServeComputationTime = perf_counter()
            _, firstComeFirstServeSchedulerDuration = firstComeFirstServe(resourceConstraints, taskToBeScheduled)
            # stop the stopwatch
            firstComeFirstServeComputationTime = perf_counter() - firstComeFirstServeComputationTime

        """ store the raw dataset in a dictionary """

        # create a dataset that contain only the length of resourceConstraints and taskToBeScheduled
        if (smallDataset is True):

            if (withComputationTime is True):

                dataset[number] = {
                    "resourceConstraints": len(resourceConstraints),
                    "taskToBeScheduled": len(taskToBeScheduled),
                    "schedulingMethod": {
                        "octasortAndFit": {
                            "computationTime": octasortAndFitComputationTime,
                            "schedulerDuration": octasortAndFitSchedulerDuration
                        },
                        "sortAndFit": {
                            "computationTime": sortAndFitComputationTime,
                            "schedulerDuration": sortAndFitSchedulerDuration
                        },
                        "shortestJobNext": {
                            "computationTime": shortestJobNextComputationTime,
                            "schedulerDuration": shortestJobNextSchedulerDuration
                        },
                        "firstComeFirstServe": {
                            "computationTime": firstComeFirstServeComputationTime,
                            "schedulerDuration": firstComeFirstServeSchedulerDuration
                        }
                    }
                }

            else:

                dataset[number] = {
                    "resourceConstraints": len(resourceConstraints),
                    "taskToBeScheduled": len(taskToBeScheduled)
                }

        # create a dataset that contain the resourceConstraints and taskToBeScheduled
        else:

            if (withComputationTime is True):

                dataset[number] = {
                    "resourceConstraints": resourceConstraints,
                    "taskToBeScheduled": taskToBeScheduled,
                    "schedulingMethod": {
                        "octasortAndFit": {
                            "computationTime": octasortAndFitComputationTime,
                            "schedulerDuration": octasortAndFitSchedulerDuration
                        },
                        "sortAndFit": {
                            "computationTime": sortAndFitComputationTime,
                            "schedulerDuration": sortAndFitSchedulerDuration
                        },
                        "shortestJobNext": {
                            "computationTime": shortestJobNextComputationTime,
                            "schedulerDuration": shortestJobNextSchedulerDuration
                        },
                        "firstComeFirstServe": {
                            "computationTime": firstComeFirstServeComputationTime,
                            "schedulerDuration": firstComeFirstServeSchedulerDuration
                        }
                    }
                }

            else:

                dataset[number] = {
                    "resourceConstraints": resourceConstraints,
                    "taskToBeScheduled": taskToBeScheduled
                }

        # update the tqdm progress bar
        progress.update(1)

    # close the tqdm progress bar
    progress.close()

    return dataset

"""" utilizing 'octasort and fit' to arrange the iot task execution order with multiple constraints into a scheduler """

# function to schedule the iot tasks based on "Octasort and Fit" concept
def octasortAndFit(resourceConstraints, taskToBeScheduled):

    # function to check if the resource to be taken by the iot task is enough or not
    # NOTE: we use nested function for speed
    def constraintChecker(resourceConstraints, resourceTaken, resourceToBeTaken):

        canBeIncluded = True

        # for every resources labelled by the iot task to be taken
        for label in resourceToBeTaken.keys():

            # if the resourse to be taken exceed the constraint
            if (resourceConstraints[label] < resourceTaken[label] + resourceToBeTaken[label]):

                canBeIncluded = False
                break

        return canBeIncluded

    # each list inside the dictionary represent 8 type of priority the iot tasks will be sorted
    sortedExecution = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

    # sort according to the runtime length in ascending order
    sortedExecution[0] = sorted(taskToBeScheduled, key=lambda x: x[1])
    # sort according to the runtime length in descending order
    sortedExecution[1] = sorted(taskToBeScheduled, key=lambda x: x[1], reverse=True)
    # sort according to the sum of resource constraints in ascending order
    sortedExecution[2] = sorted(taskToBeScheduled, key=lambda x: sum(x[2].values()))
    # sort according to the sum of resource constraints in descending order
    sortedExecution[3] = sorted(taskToBeScheduled, key=lambda x: sum(x[2].values()), reverse=True)
    # sort according to the runtime length divided by the sum of resource constraints in ascending order
    sortedExecution[4] = sorted(taskToBeScheduled, key=lambda x: (x[1] / sum(x[2].values()) if (x[1] and sum(x[2].values()) > 0) else 0))
    # sort according to the runtime length divided by the sum of resource constraints in descending order
    sortedExecution[5] = sorted(taskToBeScheduled, key=lambda x: (x[1] / sum(x[2].values()) if (x[1] and sum(x[2].values()) > 0) else 0), reverse=True)
    # sort according to the sum of resource constraints divided by the runtime length in ascending order
    sortedExecution[6] = sorted(taskToBeScheduled, key=lambda x: (sum(x[2].values()) / x[1] if (x[1] and sum(x[2].values()) > 0) else 0))
    # sort according to the sum of resource constraints divided by the runtime length in descending order
    sortedExecution[7] = sorted(taskToBeScheduled, key=lambda x: (sum(x[2].values()) / x[1] if (x[1] and sum(x[2].values()) > 0) else 0), reverse=True)

    # for storing the scheduler duration and result
    schedulerDuration = 0
    generatedScheduler = {}

    # because there are 8 type of sorting methods
    for method in range(8):

        # dictionary to record every resource taken
        resourceTaken = {}

        # initialize the dictionary according to the number of constraints to be recorded
        for constraint in resourceConstraints.keys():

            resourceTaken[constraint] = 0

        # initialize the number of iot task to be assigned
        toBeAssignedTasks = len(taskToBeScheduled)
        # to mark the iot task position in list that are currently being tested for fitting
        skippingMarker = 0
        # to track how many time the execution step has occured
        timeStep = 0
        # to store the shortest execution time
        shortestTime = 0
        # for storing the schedulerDuration temporarily before doing the comparison
        tempSchedulerDuration = 0
        # for storing the generatedScheduler temporarily before doing the comparison
        tempGeneratedScheduler = {}
        # to store the fitting result temporarily
        temp = []
        # to record the iot tasks that are currently running in simulation right now
        runningTasks = []

        # perform the fitting task as long there are remaining tasks
        while (toBeAssignedTasks != 0):

            # due to the addition or subtraction in float format could never be a 0
            for constraint in resourceConstraints.keys():

                # round off all the numbers in dictionary to a 3 decimal points
                # having an accuracy in 3 decimal points is good enough
                resourceTaken[constraint] = round(resourceTaken[constraint], 3)

            # prioritize emergency mode type of iot task first
            # as long the taskToBeScheduled is not empty
            if (toBeAssignedTasks != 0):

                # as long the next iot task in sequence does not exceed the system resource limit,
                # skippingMarker does not exceed the toBeAssignedTasks length
                # NOTE: this "while loop" section act as a fitter to see if the iot tasks can be executed or not
                while ((skippingMarker < toBeAssignedTasks) and (taskToBeScheduled != [])):

                    # check the iot task if it can be fitted or not
                    if (constraintChecker(resourceConstraints, resourceTaken, sortedExecution[method][skippingMarker][2]) is True):

                        # reduce the counter by 1 because 1 iot task is assigned
                        toBeAssignedTasks -= 1
                        # temporarily save the iot task
                        temp.append(sortedExecution[method][skippingMarker])
                        # record the iot task as being executed right now (increment the iot task's runtime according to the timeStep)
                        runningTasks.append([sortedExecution[method][skippingMarker][0], sortedExecution[method][skippingMarker][1] + timeStep, sortedExecution[method][skippingMarker][2]])

                        # record every resources consumed by the iot task
                        for constraint in sortedExecution[method][skippingMarker][2].keys():

                            resourceTaken[constraint] += sortedExecution[method][skippingMarker][2][constraint]

                        # delete the task from the list
                        del sortedExecution[method][skippingMarker]

                    # if not jump to the next iot task inside the list
                    else:

                        skippingMarker += 1

                # must reset skippingMarker to prevent out of bound list error
                skippingMarker = 0

            # if the fitter found iot task that can be executed
            if (temp != []):

                # save the iot task with the time for it to run as dictionary key
                tempGeneratedScheduler[timeStep] = temp
                # reset the temp
                temp = []

            # if there are iot tasks that are currently running
            if (runningTasks != []):

                # find the iot task that will complete their job soon
                # NOTE: the iot task that will complete their job soon has the shortest running time
                shortestTime = min(runningTasks, key=lambda i: i[1])[1]
                # record the time for the iot task that will complete their job soon
                # NOTE: at the time for the iot task that will complete their job soon, iot system resources will be freed!
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
            tempSchedulerDuration = max(runningTasks, key=lambda i: i[1])[1]

        # if the runningTasks is empty (all the tasks has been marked as completed)
        else:

            # the shortestTime mark the scheduler duration
            tempSchedulerDuration = shortestTime

        # if there are no previous comparison for the scheduler
        if (schedulerDuration == 0):

            # swap it out (no need to use copy.deepcopy() because the temp value will be replaced not edited)
            schedulerDuration = tempSchedulerDuration
            generatedScheduler = tempGeneratedScheduler

        # if the previously generated scheduler has longer duration compared to the new one
        elif (schedulerDuration > tempSchedulerDuration):

            # swap it out (no need to use copy.deepcopy() because the temp value will be replaced not edited)
            schedulerDuration = tempSchedulerDuration
            generatedScheduler = tempGeneratedScheduler

    return generatedScheduler, schedulerDuration

"""" utilizing 'sort and fit' to arrange the iot task execution order with multiple constraints into a scheduler """

# function to schedule the iot tasks based on "Sort and Fit" concept
def sortAndFit(resourceConstraints, taskToBeScheduled):

    # function to check if the resource to be taken by the iot task is enough or not
    # NOTE: we use nested function for speed
    def constraintChecker(resourceConstraints, resourceTaken, resourceToBeTaken):

        canBeIncluded = True

        # for every resources labelled by the iot task to be taken
        for label in resourceToBeTaken.keys():

            # if the resourse to be taken exceed the constraint
            if (resourceConstraints[label] < resourceTaken[label] + resourceToBeTaken[label]):

                canBeIncluded = False
                break

        return canBeIncluded

    # each list inside the dictionary represent 4 type of priority the iot tasks will be sorted
    sortedExecution = {0: [], 1: [], 2: [], 3: []}

    # sort according to the runtime length in ascending order
    sortedExecution[0] = sorted(taskToBeScheduled, key=lambda x: x[1])
    # sort according to the sum of resource constraints in ascending order
    sortedExecution[1] = sorted(taskToBeScheduled, key=lambda x: sum(x[2].values()))
    # sort according to the runtime length divided by the sum of resource constraints in ascending order
    sortedExecution[2] = sorted(taskToBeScheduled, key=lambda x: (x[1] / sum(x[2].values()) if (x[1] and sum(x[2].values()) > 0) else 0))
    # sort according to the sum of resource constraints divided by the runtime length in ascending order
    sortedExecution[3] = sorted(taskToBeScheduled, key=lambda x: (sum(x[2].values()) / x[1] if (x[1] and sum(x[2].values()) > 0) else 0))

    # for storing the scheduler duration and result
    schedulerDuration = 0
    generatedScheduler = {}

    # because there are 4 type of sorting methods
    for method in range(4):

        # dictionary to record every resource taken
        resourceTaken = {}

        # initialize the dictionary according to the number of constraints to be recorded
        for constraint in resourceConstraints.keys():

            resourceTaken[constraint] = 0

        # initialize the number of iot task to be assigned
        toBeAssignedTasks = len(taskToBeScheduled)
        # to mark the iot task position in list that are currently being tested for fitting
        skippingMarker = 0
        # to track how many time the execution step has occured
        timeStep = 0
        # to store the shortest execution time
        shortestTime = 0
        # for storing the schedulerDuration temporarily before doing the comparison
        tempSchedulerDuration = 0
        # for storing the generatedScheduler temporarily before doing the comparison
        tempGeneratedScheduler = {}
        # to store the fitting result temporarily
        temp = []
        # to record the iot tasks that are currently running in simulation right now
        runningTasks = []

        # perform the fitting task as long there are remaining tasks
        while (toBeAssignedTasks != 0):

            # due to the addition or subtraction in float format could never be a 0
            for constraint in resourceConstraints.keys():

                # round off all the numbers in dictionary to a 3 decimal points
                # having an accuracy in 3 decimal points is good enough
                resourceTaken[constraint] = round(resourceTaken[constraint], 3)

            # prioritize emergency mode type of iot task first
            # as long the taskToBeScheduled is not empty
            if (toBeAssignedTasks != 0):

                # as long the next iot task in sequence does not exceed the system resource limit,
                # skippingMarker does not exceed the toBeAssignedTasks length
                # NOTE: this "while loop" section act as a fitter to see if the iot tasks can be executed or not
                while ((skippingMarker < toBeAssignedTasks) and (taskToBeScheduled != [])):

                    # check the iot task if it can be fitted or not
                    if (constraintChecker(resourceConstraints, resourceTaken, sortedExecution[method][skippingMarker][2]) is True):

                        # reduce the counter by 1 because 1 iot task is assigned
                        toBeAssignedTasks -= 1
                        # temporarily save the iot node
                        temp.append(sortedExecution[method][skippingMarker])
                        # record the iot task as being executed right now (increment the iot task's runtime according to the timeStep)
                        runningTasks.append([sortedExecution[method][skippingMarker][0], sortedExecution[method][skippingMarker][1] + timeStep, sortedExecution[method][skippingMarker][2]])

                        # record every resources consumed by the iot task
                        for constraint in sortedExecution[method][skippingMarker][2].keys():

                            resourceTaken[constraint] += sortedExecution[method][skippingMarker][2][constraint]

                        # delete the task from the list
                        del sortedExecution[method][skippingMarker]

                    # if not jump to the next iot task inside the list
                    else:

                        skippingMarker += 1

                # must reset skippingMarker to prevent out of bound list error
                skippingMarker = 0

            # if the fitter found iot task that can be executed
            if (temp != []):

                # save the iot nodes with the time for it to run as dictionary key
                tempGeneratedScheduler[timeStep] = temp
                # reset the temp
                temp = []

            # if there are iot tasks that are currently running
            if (runningTasks != []):

                # find the iot task that will complete their job soon
                # NOTE: the iot task that will complete their job soon has the shortest running time
                shortestTime = min(runningTasks, key=lambda i: i[1])[1]
                # record the time for the iot task that will complete their job soon
                # NOTE: at the time for the iot task that will complete their job soon, iot system resources will be freed!
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
            tempSchedulerDuration = max(runningTasks, key=lambda i: i[1])[1]

        # if the runningTasks is empty (all the tasks has been marked as completed)
        else:

            # the shortestTime mark the scheduler duration
            tempSchedulerDuration = shortestTime

        # if there are no previous comparison for the scheduler
        if (schedulerDuration == 0):

            # swap it out (no need to use copy.deepcopy() because the temp value will be replaced not edited)
            schedulerDuration = tempSchedulerDuration
            generatedScheduler = tempGeneratedScheduler

        # if the previously generated scheduler has longer duration compared to the new one
        elif (schedulerDuration > tempSchedulerDuration):

            # swap it out (no need to use copy.deepcopy() because the temp value will be replaced not edited)
            schedulerDuration = tempSchedulerDuration
            generatedScheduler = tempGeneratedScheduler

    return generatedScheduler, schedulerDuration

"""" utilizing 'shortest job next' to arrange the iot task execution order with multiple constraints into a scheduler """

# function to schedule the iot tasks based on "Shortest Job Next" concept
def shortestJobNext(resourceConstraints, taskToBeScheduled):

    # function to check if the resource to be taken by the iot task is enough or not
    # NOTE: we use nested function for speed
    def constraintChecker(resourceConstraints, resourceTaken, resourceToBeTaken):

        canBeIncluded = True

        # for every resources labelled by the iot task to be taken
        for label in resourceToBeTaken.keys():

            # if the resourse to be taken exceed the constraint
            if (resourceConstraints[label] < resourceTaken[label] + resourceToBeTaken[label]):

                canBeIncluded = False
                break

        return canBeIncluded

    # sort according to the runtime length in ascending order
    sortedExecution = sorted(taskToBeScheduled, key=lambda x: x[1])

    # for storing the scheduler duration and result
    schedulerDuration = 0
    generatedScheduler = {}

    # dictionary to record every resource taken
    resourceTaken = {}

    # initialize the dictionary according to the number of constraints to be recorded
    for constraint in resourceConstraints.keys():

        resourceTaken[constraint] = 0

    # initialize the number of iot task to be assigned
    toBeAssignedTasks = len(taskToBeScheduled)
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
        for constraint in resourceConstraints.keys():

            # round off all the numbers in dictionary to a 3 decimal points
            # having an accuracy in 3 decimal points is good enough
            resourceTaken[constraint] = round(resourceTaken[constraint], 3)

        # as long the toBeAssignedTasks is not empty
        while (toBeAssignedTasks != 0):

            # check the iot task if it can be fitted or not
            if (constraintChecker(resourceConstraints, resourceTaken, sortedExecution[0][2]) is True):

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

            # save the iot task with the time for it to run as dictionary key
            generatedScheduler[timeStep] = temp
            # reset the temp
            temp = []

        # if there are iot tasks that are currently running
        if (runningTasks != []):

            # find the iot task that will complete their job soon
            # NOTE: the iot task that will complete their job soon has the shortest running time
            shortestTime = min(runningTasks, key=lambda i: i[1])[1]
            # record the time for the iot task that will complete their job soon
            # NOTE: at the time for the iot task that will complete their job soon, iot system resources will be freed!
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

    return generatedScheduler, schedulerDuration

"""" utilizing 'first come first serve' to arrange the iot task execution order with multiple constraints into a scheduler """

# function to schedule the iot tasks based on "First Come First Serve" concept
def firstComeFirstServe(resourceConstraints, taskToBeScheduled):

    # function to check if the resource to be taken by the iot task is enough or not
    # NOTE: we use nested function for speed
    def constraintChecker(resourceConstraints, resourceTaken, resourceToBeTaken):

        canBeIncluded = True

        # for every resources labelled by the iot task to be taken
        for label in resourceToBeTaken.keys():

            # if the resourse to be taken exceed the constraint
            if (resourceConstraints[label] < resourceTaken[label] + resourceToBeTaken[label]):

                canBeIncluded = False
                break

        return canBeIncluded

    # for storing the scheduler duration and result
    schedulerDuration = 0
    generatedScheduler = {}

    # create the taskToBeScheduled duplicate (if not, the content inside the original list will get deleted)
    tempTaskToBeScheduled = deepcopy(taskToBeScheduled)

    # dictionary to record every resource taken
    resourceTaken = {}

    # initialize the dictionary according to the number of constraints to be recorded
    for constraint in resourceConstraints.keys():

        resourceTaken[constraint] = 0

    # initialize the number of iot task to be assigned
    toBeAssignedTasks = len(tempTaskToBeScheduled)
    # to track how many time the execution step has occured
    timeStep = 0
    # to store the shortest execution time
    shortestTime = 0
    # for storing the generatedScheduler temporarily before doing the comparison
    tempGeneratedScheduler = {}
    # to store the fitting result temporarily
    temp = []
    # to record the iot tasks that are currently running in simulation right now
    runningTasks = []

    # perform the fitting task as long there are remaining tasks
    while (toBeAssignedTasks != 0):

        # due to the addition or subtraction in float format could never be a 0
        for constraint in resourceConstraints.keys():

            # round off all the numbers in dictionary to a 3 decimal points
            # having an accuracy in 3 decimal points is good enough
            resourceTaken[constraint] = round(resourceTaken[constraint], 3)

        # as long the toBeAssignedTasks is not empty
        while (toBeAssignedTasks != 0):

            # check the iot task if it can be fitted or not
            if (constraintChecker(resourceConstraints, resourceTaken, tempTaskToBeScheduled[0][2]) is True):

                # reduce the counter by 1 because 1 iot task is assigned
                toBeAssignedTasks -= 1
                # temporarily save the iot node
                temp.append(tempTaskToBeScheduled[0])
                # record the iot task as being executed right now (increment the iot task's runtime according to the timeStep)
                runningTasks.append([tempTaskToBeScheduled[0][0], tempTaskToBeScheduled[0][1] + timeStep, tempTaskToBeScheduled[0][2]])

                # record every resources consumed by the iot task
                for constraint in tempTaskToBeScheduled[0][2].keys():

                    resourceTaken[constraint] += tempTaskToBeScheduled[0][2][constraint]

                # delete the task from the list
                del tempTaskToBeScheduled[0]

            else:

                # stop looking (for now)
                break

        # if the fitter found iot task that can be executed
        if (temp != []):

            # save the iot task with the time for it to run as dictionary key
            generatedScheduler[timeStep] = temp
            # reset the temp
            temp = []

        # if there are iot tasks that are currently running
        if (runningTasks != []):

            # find the iot task that will complete their job soon
            # NOTE: the iot task that will complete their job soon has the shortest running time
            shortestTime = min(runningTasks, key=lambda i: i[1])[1]
            # record the time for the iot task that will complete their job soon
            # NOTE: at the time for the iot task that will complete their job soon, iot system resources will be freed!
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

    return generatedScheduler, schedulerDuration

# initiator
if __name__ == "__main__":

    os.system("cls")
    # because it will affect the dataset creation time measurement
    print(Fore.RED + "\n1. Make sure that you are launching this simulation program without any background tasks running!")
    print(Fore.RED + "2. The next programs must run on the same computer to ensure reproducibility, otherwise the experiment is invalid!" + Style.RESET_ALL)
    # unlabeled dataset filename
    unlabeledDatasetName = input("\nEnter the unlabeled dataset's name you want to save in 01_dataset folder (.pickle): ")
    unlabeledDatasetName = "./01_dataset/" + unlabeledDatasetName
    # create a dataset that contain 25,000 unlabeled data
    cycle = sVS.cycle
    # randomization key to make the dataset reproducible
    seedKey = int(input("Enter the randomization key: "))
    # probability for the resource constraint consumed by the device not to become zero
    flipCoinBias = sVS.flipCoinBias

    """
    randomization range for the number of tasks: 3 ~ 100
    randomization range for the number of resource constraints: 1 ~ 10
    randomization range for the resource constraint value: 10 ~ 100

    """

    # randomization range for the number of iot tasks in a system
    numberOfTasks_range = sVS.numberOfTasks_range
    # randomization range for the value of resource constraint in a system
    valueOfResourceConstraints_range = sVS.valueOfResourceConstraints_range
    # randomization range for the number of resource constraint in a system
    numberOfResourceConstraints_range = sVS.numberOfResourceConstraints_range

    # create a small dataset to train the ai
    smallDataset = True
    # include the computation time taken for each scheduling algorithm
    withComputationTime = True

    print("\nGenerating an unlabeled dataset to train the AI.")

    # if dataset folder does not exist
    if not os.path.exists("01_dataset"):

        # create the dataset folder
        os.makedirs("01_dataset")

    # call the function with pre-setted parameters
    dataset = datasetRandomGenerator(
        cycle,
        seedKey,
        flipCoinBias,
        smallDataset,
        withComputationTime,
        numberOfTasks_range,
        valueOfResourceConstraints_range,
        numberOfResourceConstraints_range
    )

    """ save the dictionary as a file in pickle format """

    # create and open the file
    pickleOut = open(unlabeledDatasetName, "wb")
    # write the dataset into the file
    pickle.dump(dataset, pickleOut)
    # close the file
    pickleOut.close()

    print("\nRaw dataset was succesfully generated!")
