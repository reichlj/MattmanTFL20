import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow Version',tf.__version__)

def model(X, w):
    return tf.multiply(X, w)

learning_rate = 0.01
training_epochs = 100
sample_size = 101
x_train = np.linspace(-1, 1, sample_size)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(0.0, name="weights")
y_model = model(X, w)
cost = tf.square(Y-y_model)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# training
for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X: x, Y: y})
w_val = sess.run(w)
sess.close()

print('w_val',w_val)
plt.scatter(x_train, y_train)
y_learned = x_train*w_val
plt.plot(x_train, y_learned, 'r')
plt.title('w={0:0.4f}   sample_size={1}   epochs={2}'.format(
           w_val, sample_size, training_epochs))
plt.show()