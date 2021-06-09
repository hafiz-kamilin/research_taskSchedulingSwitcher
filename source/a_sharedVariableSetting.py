#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Mohd Hafizuddin Bin Kamilin"
__version__ = "1.0.10"

""" variables for controlling the dataset creation """

# create a dataset that contain 25,000 unlabeled data
cycle = 25000
# probability for the resource consumed by the device not to become zero
flipCoinBias = 80
# randomization range for the number of iot tasks in a system
numberOfTasks_range = [3, 100]
# randomization range for the number of resource constraint in a system
numberOfResourceConstraints_range = [1, 10]
# randomization range for the value of resource constraint in a system
valueOfResourceConstraints_range = [10, 100]

""" variables for controlling the dataset normalization """

# maximum computation deadline [second]
maxComputationDeadline = 1

""" variables for controlling the dataset creation for switch """

# create a dataset that contain 2,500 unlabeled data
cycleSwitch = 2500
# number of scheduling algorithms used in this experiment
numberOfAlgorithm = 4

""" variable for controlling the simulation """

# interfacing latency of the MTL model
latencyH5 = 0.005599596227000182
latencyTflite = 0.000030891416000008576
# loop number for the scheduler to be generated in 1 hour
hourCycle = 1000
# NOTE: for the sake of easy debugging, we set the first 5 schedulers in 1 hours to be shown
hourSample = 5
