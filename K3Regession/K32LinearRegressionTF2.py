import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow Version',tf.__version__)

learning_rate = 0.01
training_epochs = 50
sample_size = 101
x_train = np.linspace(-1, 1, sample_size)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

X = tf.constant(x_train, dtype=tf.float32)
Y = tf.constant(y_train, dtype=tf.float32)
w = tf.Variable(0., name="weights", dtype=tf.float32)
cost = lambda: tf.square(Y - tf.multiply(X, w))

train_op = tf.keras.optimizers.SGD(learning_rate)
for _ in range(training_epochs):
    train_op.minimize(cost, w)
    print(w.numpy())

val1 = w.value()
print("Prediction: {}".format(val1))

plt.scatter(x_train, y_train)
y_learned = x_train * val1
plt.plot(x_train, y_learned, 'r')
plt.title('w={0:0.4f}   sample_size={1}   epochs={2}'.format(
           val1, sample_size, training_epochs))
plt.show()

