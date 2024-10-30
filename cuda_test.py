import tensorflow as tf
from tensorflow.python.client import device_lib

# Check if the GPU and CuDNN are recognized
print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
