import tensorflow as tf
import numpy as np
import gzip

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
VALIDATION_SIZE = 5000
BATCH_SIZE = 64

def extract_data(file_name, num_images):
    print('extract %s' % file_name)
    with gzip.open(file_name) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE*IMAGE_SIZE*num_images*NUM_CHANNELS)
        # test as float64
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
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

    input_data = tf.placeholder(tf.float64,
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    input_lable = tf.placeholder(tf.int64,
            shape=(BATCH_SIZE))
    eval_data = tf.placeholder(tf.float64,
            shape=(EVAL_BATCH, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

if __name__ == '__main__':
    tf.app.run(main=main)
