#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午7:47
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : write_text_features.py
# @IDE: PyCharm Community Edition
"""
Write text features into tensorflow records
"""
import os
import os.path as ops
import argparse
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import sys
#sys.path.append('/data2/hdia_ocr_data/CRNN')
sys.path.append('/data2/hdia_ocr_data/CRNN/data_provider')
import data_provider
sys.path.append('/data2/hdia_ocr_data/CRNN/local_utils')
import data_utils
#from data_provider import data_provider
#from local_utils import data_utils


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--save_dir', type=str, help='Where you store tfrecords')

    return parser.parse_args()


def write_features(dataset_dir, save_dir):
    """

    :param dataset_dir:
    :param save_dir:
    :return:
    """
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    print('Initialize the dataset provider ......')
    provider = data_provider.TextDataProvider(dataset_dir=dataset_dir, annotation_name='sample.txt',
                                              validation_set=True, validation_split=0.12505, shuffle='every_epoch',
                                              normalization=None)
    print('Dataset provider intialize complete')

    feature_io = data_utils.TextFeatureIO()

    # write train tfrecords
    print('Start writing training tf records')

    train_images = provider.train.images; train_imagenames = provider.train.imagenames
    print(len(train_images))
    train_images = [cv2.resize(tmp, (600, 32)) for tmp in train_images]
    '''train_images1 = [];
    cnt = 0; cntlist = [];
    for tmp in train_images:
        try: 
            train_images1.append(cv2.resize(tmp, (100, 32)));
            cntlist.append(cnt); #print("done: ",cnt);
        except: print(cnt,train_imagenames[cnt],"here")
        cnt +=1;
    print("done: ",cnt);'''	
    train_images = [bytes(list(np.reshape(tmp, [600 * 32 * 3]))) for tmp in train_images]
    train_labels = provider.train.labels
    '''cnt = 0;
    for i in provider.train.labels:
        if cnt in cntlist: train_labels.append(i)
        cnt +=1;'''
    train_imagenames = provider.train.imagenames

    train_tfrecord_path = ops.join(save_dir, 'train_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=train_tfrecord_path, labels=train_labels, images=train_images,
                                     imagenames=train_imagenames)

    # write test tfrecords
    print('Start writing testing tf records')

    test_images = provider.test.images
    test_images = [cv2.resize(tmp, (600, 32)) for tmp in test_images]
    '''test_images1 = [];
    cnt = 0; cntlist = [];
    for tmp in test_images:
        try: 
            test_images1.append(cv2.resize(tmp, (100, 32)));
            cntlist.append(cnt);
        except: pass
        cnt +=1;'''	
    test_images = [bytes(list(np.reshape(tmp, [600 * 32 * 3]))) for tmp in test_images]
    test_labels = provider.test.labels
    test_imagenames = provider.test.imagenames

    test_tfrecord_path = ops.join(save_dir, 'test_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=test_tfrecord_path, labels=test_labels, images=test_images,
                                     imagenames=test_imagenames)

    # write val tfrecords
    print('Start writing validation tf records')

    val_images = provider.validation.images
    '''val_images1 = [];
    cnt = 0; cntlist = [];
    for tmp in val_images:
        try: 
            val_images1.append(cv2.resize(tmp, (100, 32)));
            cntlist.append(cnt);
        except: pass
        cnt +=1;'''	
    val_images = [cv2.resize(tmp, (600, 32)) for tmp in val_images]
    val_images = [bytes(list(np.reshape(tmp, [600 * 32 * 3]))) for tmp in val_images]
    val_labels = provider.validation.labels
    val_imagenames = provider.validation.imagenames

    val_tfrecord_path = ops.join(save_dir, 'validation_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=val_tfrecord_path, labels=val_labels, images=val_images,
                                     imagenames=val_imagenames)

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    # write tf records
    write_features(dataset_dir=args.dataset_dir, save_dir=args.save_dir)
