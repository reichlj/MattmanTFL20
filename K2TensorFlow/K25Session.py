import tensorflow as tf
print('Version',tf.__version__)

x = tf.constant([[1., 2.]])
negMatrix = tf.negative(x)
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
    options = tf.compat.v1.RunOptions(output_partition_graphs=True)
    metadata = tf.compat.v1.RunMetadata()
    result = sess.run(negMatrix,options=options, run_metadata=metadata)
print('Result',result)
print('MetaData',metadata.partition_graphs)
