import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

raw_data = np.random.normal(10, 1, 100)
alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg
init = tf.global_variables_initializer()
sess.run(init)
for i in range(len(raw_data)):
    # curr_avg = sess.run(update_avg, feed_dict={curr_value: raw_data[i]})
    curr_avg = sess.run(update_avg, feed_dict={curr_value: raw_data[i]})

    # sess.run(tf.assign(prev_avg, curr_avg))
    updater = tf.assign(prev_avg, curr_avg)
    updater.eval()
    print(raw_data[i], curr_avg)
sess.close()