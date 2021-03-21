import tensorflow as tf
print('Version',tf.__version__)
# create a constant Tensor object
x = tf.constant([[1.0, 2.0]])
negMatrix = tf.negative(x)
print(negMatrix)
