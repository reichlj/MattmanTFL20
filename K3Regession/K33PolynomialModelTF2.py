import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow Version',tf.__version__)

learning_rate = 0.01
training_epochs = 40
sample_size = 101
# create training data - numpy data
trX = np.linspace(-1, 1, sample_size)
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(len(trY_coeffs)):
    trY += trY_coeffs[i] * np.power(trX, i)
trY += np.random.randn(*trX.shape) * 1.5

X = tf.constant(trX, dtype=tf.float32)
Y = tf.constant(trY, dtype=tf.float32)
w = tf.Variable([0.] * len(trY_coeffs), name="parameters")

model = lambda _X, _w: tf.add_n([tf.multiply(_w[i], tf.pow(_X, i))
                                 for i in range(len(trY_coeffs))])
y_model = lambda: model(X, w)
cost = lambda: tf.pow(Y - y_model(), 2)

train_op = tf.keras.optimizers.SGD(learning_rate, momentum=0.5)
for epoch in range(training_epochs):
    train_op.minimize(cost,w)

w_val = w.value()
print(w_val.numpy())

plt.scatter(trX, trY)
trY2 = 0
for i in range(len(trY_coeffs)):
    trY2 += w_val[i] * np.power(trX, i)

plt.plot(trX, trY2, 'r')
plt.title('w0={0:0.2f} w1={1:0.2f}   sample_size={2}   epochs={3}'.format(
           w_val[0], w_val[1], sample_size, training_epochs))
plt.show()