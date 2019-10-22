import tensorflow as tf
import numpy as np
import gzip
import time
import sys

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000
BATCH_SIZE = 64
EVAL_BATCH = 64
NUM_LABELS = 10
EVAL_FREQUENCE = 10

def data_type():
    return np.float64

def extract_data(file_name, num_images):
    print('extract %s' % file_name)
    with gzip.open(file_name) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE*IMAGE_SIZE*num_images*NUM_CHANNELS)
        # test as float64
        data = np.frombuffer(buf, dtype=np.uint8).astype(data_type())
        data = (data-(PIXEL_DEPTH/2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_label(file_name, num_images):
    print('extract label %s' % file_name)
    with gzip.open(file_name) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return data

def error_rate(prediction, label):
    return 100.0 - 100.0 * np.sum(
            np.argmax(prediction, 1)==label)/prediction.shape[0]

def main(_):
    train_data_filename = 'train-images-idx3-ubyte.gz'
    train_labels_filename = 'train-labels-idx1-ubyte.gz'
    test_data_filename = 't10k-images-idx3-ubyte.gz'
    test_labels_filename = 't10k-labels-idx1-ubyte.gz'

    train_data = extract_data(train_data_filename, 60000)
    train_label = extract_label(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_label = extract_label(test_labels_filename, 10000)

    validation_data = train_data[:VALIDATION_SIZE]
    validation_label = train_label[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:]
    train_label = train_label[VALIDATION_SIZE:]
    num_epoch = 10

    train_size = train_label.shape[0]

    input_data_node = tf.placeholder(data_type(),
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    input_label_node = tf.placeholder(tf.int64,
            shape=(BATCH_SIZE))
    eval_data_node = tf.placeholder(data_type(),
            shape=(EVAL_BATCH, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
        stddev=0.1, dtype=data_type()))
    conv1_bias = tf.Variable(tf.zeros([32], dtype=data_type()))
    fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE // 4 * 32, NUM_LABELS], stddev=0.1, dtype = data_type()))
    fc1_bias = tf.Variable(tf.zeros([NUM_LABELS], dtype=data_type()))

    def model(data, train=False):
        conv1 = tf.nn.conv2d(data, conv1_weights, 
                strides=[1, 1, 1, 1],
                padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        pool1 = tf.nn.max_pool(relu, 
                ksize = [1,2,2,1],
                strides=[1,2,2,1],
                padding='SAME')
        pool_shape = pool1.get_shape().as_list()
        reshape = tf.reshape(pool1,
                [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
        return tf.matmul(reshape, fc1_weights) + fc1_bias

    logits = model(input_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_label_node, logits=logits))

    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_bias))
    loss += 5e-4 * regularizers

    batch = tf.Variable(0, dtype=data_type())
    learning_rate = tf.train.exponential_decay(
            0.01,                # base rate
            batch * BATCH_SIZE,  # current index
            train_size,
            0.95,                # decay rate
            staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(
            loss, global_step=batch)

    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model(eval_data_node))

    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH:
            raise ValueError("batch size larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float64)
        for begin in range(0, size, EVAL_BATCH):
            end = begin + EVAL_BATCH
            if end <= size:
                predictions[begin:end, :] = sess.run(
                        eval_prediction,
                        feed_dict={eval_data_node: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                        eval_prediction,
                        feed_dict={eval_data_node: data[-EVAL_BATCH:, ...]})
                predictions[begin:, :] = batch_predictions[begin-size:, :]
        return predictions

    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('init')
        for step in range(int(num_epoch * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_label = train_label[offset:(offset+BATCH_SIZE)]
            feed_dict = {input_data_node: batch_data,
                    input_label_node: batch_label}
            sess.run(optimizer, feed_dict=feed_dict)
            if step % EVAL_FREQUENCE == 0:
                l, lr, predictions = sess.run([loss, learning_rate, 
                    train_prediction], feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                print('step %d (epoch %.2f), %.1f ms' % (step, 
                    float(step)*BATCH_SIZE / train_size,
                    1000 * elapsed_time / EVAL_FREQUENCE))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_label))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_label))
                sys.stdout.flush()

        test_error = error_rate(eval_in_batches(test_data, sess), test_label)
        print('Test error: %.1f%%' % test_error)
        

if __name__ == '__main__':
    tf.app.run(main=main)
