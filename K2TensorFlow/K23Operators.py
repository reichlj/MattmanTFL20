import tensorflow as tf
print('Version',tf.__version__)
# create a constant Tensor object
x = tf.constant([[1.0, 2.0]])
negMatrix = tf.negative(x)
print(negMatrix)

x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)
with tf.Session() as sess:
    result = sess.run(neg_op)
print(result,type(result))

sess = tf.InteractiveSession()
x = tf.constant([[1., 2.]])
negMatrix = tf.negative(x)
result = negMatrix.eval()
print(result,type(result))
sess.close()
