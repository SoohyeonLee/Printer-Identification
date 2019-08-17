import tensorflow as tf
import numpy as np

import os
import sys

from urllib.request import urlopen

#Error 출력 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Setting

PATH = "./"
MODEL_PATH = "D:" + os.sep + "DeepLearning" + os.sep + "result"
PATH_DATASET = PATH + os.sep + "dataset_100"
FILE_TRAIN = PATH_DATASET + os.sep + "train.tfrecords"
FILE_TEST = PATH_DATASET + os.sep + "test.tfrecords"

TRAIN_EXAMPLES_NUM = 30000*8
TEST_EXAMPLES_NUM = 7200*8

BATCH_SIZE = 120
STEPS_PER_EPOCH = TRAIN_EXAMPLES_NUM/BATCH_SIZE
MAX_EPOCH = 300
TOTAL_CHECKPOINT = 1000
SAVE_CHECKPOINT = STEPS_PER_EPOCH * 10

CLASS_NUM = 8
DROP_OUT = 0.5


def print_activations(tensor):
    print(tensor.op.name, ' ', tensor.get_shape().as_list())

def soo_input_fn(file_path, perform_shuffle=False, repeat_count=1):

    def decode(serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        
        image = tf.decode_raw(features['image'], tf.uint8)
        
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'],tf.int64) 
    
        return dict({'image':image}),label

    dataset = tf.data.TFRecordDataset(file_path).map(decode)
    
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256,reshuffle_each_iteration=False)
    
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_label = iterator.get_next()

    return batch_features, batch_label

def soo_model_fn(features, labels, mode):    

    feature_columns = [tf.feature_column.numeric_column(key='image',shape=(128*128*3))]

    input_layer = tf.feature_column.input_layer(features, feature_columns)
    input_layer = tf.reshape(input_layer,[-1,128,128,3])

    
    F0 = np.array([[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]], dtype=np.float32)
    F1 = F0 / 12.

    Filter = np.array(F1, dtype=np.float32)
    my_filter = tf.constant_initializer(value=Filter, dtype=tf.float32)

    HPF = tf.layers.Conv2D(
        filters=1, 
        activation=None, 
        kernel_size=5, 
        trainable=False, 
        strides=1, 
        padding="SAME", 
        use_bias = False, 
        kernel_initializer=my_filter)(input_layer)
    

    with tf.name_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(HPF, 32, 3, strides=1, padding='SAME', activation=tf.nn.relu, name=scope)
        bn1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True)
        print_activations(conv1)

    with tf.name_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(bn1, 32, 3, strides=1, padding='SAME', activation=tf.nn.relu, name=scope)
        bn2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True)
        print_activations(conv2)

    with tf.name_scope('conv3') as scope:
        conv3 = tf.layers.conv2d(bn2, 32, 3, strides=1, padding='SAME', activation=tf.nn.relu, name=scope)
        bn3 = tf.contrib.layers.batch_norm(conv3, center=True, scale=True)
        print_activations(conv3)

    pool1 = tf.layers.max_pooling2d(bn3, pool_size=3, strides=2, padding='SAME', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv4') as scope:
        conv4 = tf.layers.conv2d(pool1, 64, 3, strides=1, padding='SAME', activation=tf.nn.relu, name=scope)
        bn4 = tf.contrib.layers.batch_norm(conv4, center=True, scale=True)
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        conv5 = tf.layers.conv2d(bn4, 64, 3, strides=1, padding='SAME', activation=tf.nn.relu, name=scope)
        bn5 = tf.contrib.layers.batch_norm(conv5, center=True, scale=True)
        print_activations(conv5)

    with tf.name_scope('conv6') as scope:
        conv6 = tf.layers.conv2d(bn5, 64, 3, strides=1, padding='SAME', activation=tf.nn.relu, name=scope)
        bn6 = tf.contrib.layers.batch_norm(conv6, center=True, scale=True)
        print_activations(conv6)

    pool2 = tf.layers.max_pooling2d(bn6, pool_size=3, strides=2, padding='SAME', name='pool2')
    print_activations(pool2)


    flatten = tf.layers.flatten(pool2)
    

    with tf.name_scope('fc1') as scope:
        fc1 = tf.layers.Dense(4096, activation=tf.nn.relu, name=scope)(flatten)
        print_activations(fc1)

    drop1 = tf.layers.dropout(fc1, rate=DROP_OUT, name='drop1')
    print_activations(drop1)


    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.Dense(4096, activation=tf.nn.relu, name=scope)(drop1)
        print_activations(fc2)

    drop2 = tf.layers.dropout(fc2, rate=DROP_OUT, name='drop2')
    print_activations(drop2)


    logits = tf.layers.Dense(CLASS_NUM)(drop2)
    print_activations(logits)

    predictions = {'class_ids':tf.argmax(input=logits, axis=1) }


    # 1. Prediction mode
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    # 2. Evaluation mode
    # Return our loss (which is used to evaluate our model)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={'my_accuracy': accuracy})

    # If mode is not PREDICT nor EVAL, then we must be in TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    # 3. Training mode
    # Return training operations: loss and train_op
    learning_rate = 10e-6 * pow(0.9, tf.train.get_global_step() / (10*int(STEPS_PER_EPOCH)))
    train_op = (
            tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
                .minimize(loss, global_step=tf.train.get_global_step())
        )
        
    tf.summary.scalar('my_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# main

classifier = tf.estimator.Estimator(
    model_fn=soo_model_fn,
    model_dir=MODEL_PATH,
    config=tf.estimator.RunConfig(save_checkpoints_steps=SAVE_CHECKPOINT,keep_checkpoint_max=TOTAL_CHECKPOINT))

classifier.train(input_fn=lambda:soo_input_fn(FILE_TRAIN, True, MAX_EPOCH))

accuracy_score = classifier.evaluate(input_fn=lambda:soo_input_fn(FILE_TEST))["my_accuracy"]

print("\nTest Accuracy: {0:.2f}\n".format(accuracy_score*100))

for model in range(0,120001,2000):

    target = 'D:/DeepLearning/result/model.ckpt-'+str(model)

    accuracy_score = classifier.evaluate(input_fn=lambda:soo_input_fn(FILE_TEST),checkpoint_path=target)["my_accuracy"]

    print(str(model)+"Test Accuracy: {0:.2f}\n".format(accuracy_score*100))
