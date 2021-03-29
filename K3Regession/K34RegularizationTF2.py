import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def split_dataset(x_dataset, y_dataset, ratio):
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = int(ratio * x_dataset.size)
    x_train = x_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]]
    y_train = y_dataset[arr[0:num_train]]
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test

@tf.function
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

@tf.function
def y_model():
    return model(X, w)

@tf.function
def cost():
    w_sqr = tf.square(w)
    w_red = tf.reduce_sum(w_sqr)
    m_lmd_w_sqr = tf.multiply(tf.convert_to_tensor(reg_lambda, dtype=tf.float32), w_red)
    y_y_model_sqr = tf.reduce_sum(tf.square(Y-y_model()))
    sum_y_model_sqr_reglambda_w_sqr = tf.add(y_y_model_sqr, m_lmd_w_sqr)
    return tf.math.divide(sum_y_model_sqr_reglambda_w_sqr, 2*x_train.size)

learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.
x_dataset = np.linspace(-1, 1, 100)
num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0
for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)
# input and output
X = tf.constant(x_train, dtype=tf.float32)
Y = tf.constant(y_train, dtype=tf.float32)

w = tf.Variable([0.] * num_coeffs, name="parameters")

train_op = tf.keras.optimizers.SGD(learning_rate, momentum=0.5)

iters = 0
for reg_lambda in np.linspace(0, 1, 100):
    iters += 1
    for epoch in range(training_epochs):
        train_op.minimize(cost, w)
    final_cost = w.value()

    # Print the first 8 attempts (for illustration)
    if iters in range(9):
        plt.scatter(x_train, y_train)

        y1 = 0
        for i in range(num_coeffs):
            y1 += final_cost[i] * np.power(X, i)

        plt.plot(x_train, y1, 'r')
        plt.show()

        print('reg lambda', reg_lambda)
        print('final cost', final_cost)
