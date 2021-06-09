from os import system
system("cls")

import tensorflow as tf

cpu = tf.config.list_physical_devices(
    device_type="CPU"
)
gpu = tf.config.list_physical_devices(
    device_type="GPU"
)

print("\nNumber of hardware(s) capable to run TensorFlow " + str(tf.__version__))
print(" - CPU: " + str(len(cpu)))
print(" - GPU: " + str(len(gpu)))
