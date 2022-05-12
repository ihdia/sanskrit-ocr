from __future__ import absolute_import

import logging
import re

import tensorflow as tf

from six import b

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):

    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0

    with open(annotations_path, 'r') as annotations:
        books = ["nirnaya","kshema","kavyaprakasha"]
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image paths
            line_match = re.match(r'(\S+)\s(.*)', line)
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups()
            f=0
            for book in books:
                if img_path.find(book)!=-1:
                    f=1
                    break
            if f==0:
                try: 
                    data = plt.imread(img_path)
                    rows , columns = np.where(data < 128)

                    roi = data[min(rows):max(rows)+1, min(columns):max(columns)+1]
                    im = Image.fromarray(roi)
                    im.save(img_path)
                except:
                    print("file not found")

            try:
                with open(img_path, 'rb') as img_file:
                    img = img_file.read()

                if force_uppercase:
                    label = label.upper()

                if len(label) > len(longest_label):
                    longest_label = label
                label= [ord(str(c)) for c in label ]
                label=''.join(map(str,label))
                feature = {}
                feature['image'] = _bytes_feature(img)
                feature['label'] = _bytes_feature(b(label))
                if save_filename:
                    feature['comment'] = _bytes_feature(b(img_path))

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                writer.write(example.SerializeToString())

                if idx % log_step == 0:
                    logging.info('Processed %s pairs.', idx+1)
            except:
                print("File Not Found")

    if idx:
        logging.info('Dataset is ready: %i pairs.', idx+1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    writer.close()
