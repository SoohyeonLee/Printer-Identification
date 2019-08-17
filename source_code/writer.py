# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
import tensorflow as tf
from PIL import Image
from random import shuffle

#Writer Code

def create_record(srcDir):

    classes = os.listdir(srcDir)

    print(classes)

    file_list = []

    for _, name in enumerate(classes): #Data folder / Each Data

        class_path = srcDir + os.sep + name + os.sep

        class_list = os.listdir(class_path)
        
        for idx in range(len(class_list)): #Each Data / image

            img_name = class_list[idx]

            img_path = name + os.sep + img_name

            file_list.append(img_path)
        
    shuffle(file_list)

    #Create TFRecord file
    train_writer = tf.python_io.TFRecordWriter("./train.tfrecords")
    test_writer = tf.python_io.TFRecordWriter("./test.tfrecords")

    print("file_list len : " + str(len(file_list)))

    for idx in range(len(file_list)):

        label, filename  = file_list[idx].split('\\')

        img_path = srcDir + os.sep + label + os.sep + filename

        img = Image.open(img_path)
        
        img_byte = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_byte]))
        }))

        if idx < 240000:
            train_writer.write(example.SerializeToString())
            print('{:s} {:s} {:d} {:s}'.format(label, filename, idx, 'Train'))
        else:
            test_writer.write(example.SerializeToString())
            print('{:s} {:s} {:d} {:s}'.format(label, filename, idx, 'Test'))

    train_writer.close()
    test_writer.close()

create_record('./result')