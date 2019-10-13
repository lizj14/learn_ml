import tensorflow as tf
import numpy as np
import gzip

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000
BATCH_SIZE = 64
EVAL_BATCH = 64
NUM_LABELS = 10

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

    input_data = tf.placeholder(data_type(),
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    input_lable = tf.placeholder(tf.int64,
            shape=(BATCH_SIZE))
    eval_data = tf.placeholder(data_type(),
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
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool,
                [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
        return tf.matmul(reshape, fc1_weights) + fc1_bias

if __name__ == '__main__':
    tf.app.run(main=main)
