#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse

#from crnn_model import crnn_model
import sys
#sys.path.append('/data2/hdia_ocr_data/CRNN')
sys.path.append(os.getcwd()+'/model/CRNN/crnn_model')
sys.path.append(os.getcwd()+'/model/CRNN/local_utils')
sys.path.append(os.getcwd()+'/model/CRNN/global_configuration')
import crnn_model
import data_utils, log_utils
import config
#from local_utils import data_utils, log_utils
#from global_configuration import config


logger = log_utils.init_logger()

def get_real_median(v):
    v = tf.reshape(v, [-1])
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])

def true_fn():
    return tf.constant(0.0,dtype=tf.float32)

def false_fn1(num_nonzeros, words_len_final):
    return tf.divide(tf.cast(tf.reduce_sum(words_len_final),tf.float32), tf.cast(num_nonzeros, tf.float32))

def false_fn2(w, median):
    return tf.divide(w,median)

def init_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Dataset file')
    parser.add_argument('--weights_path', type=str, help='Where you store the pretrained weights')
    parser.add_argument('--train_epochs', type=str, help='For How Many Epochs you want to train for')
    parser.add_argument('--steps_per_checkpoint', type=str, help='Step difference between which checkpoint to be saved')

    return parser.parse_args()


def train_shadownet(filename, train_epochs, weights_path=None, steps_per_checkpoint=None):
    """
    :param dataset_dir:
    :param weights_path:
    :return:
    """
    train_epochs = int(train_epochs)
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader
    images, labels, imagenames = decoder.read_features(os.getcwd()+"/model/CRNN/data/tfReal/"+filename,num_epochs=None)
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=32, capacity=1000+2*32, min_after_dequeue=100, num_threads=1)
    
    inputdata = tf.cast(x=inputdata, dtype=tf.float32)
    #ilabels = tf.sparse_tensor_to_dense(input_labels)
    #labels = tf.convert_to_tensor(labels)
    #labels = tf.convert_to_tensor(labels)
    #labels = tf.cast(labels,dtype=tf.float32)
    #print("ZZZ"+str(ilabels))
    #ilabels = tf.Print(ilabels,[ilabels],"labelss",summarize =  1000)
    #word_av_list = []
    #w = []
    #for i in range(32):
    #    label = tf.gather_nd(ilabels, [i])
    #    label = tf.Print(label,[label],"label",summarize =  1000)
    #    element = tf.constant([32])
    #    cols = tf.where(tf.equal(label, element))[:,-1]
    #    #cols = tf.Print(cols, [cols], "cols",summarize=1000)
    #    extra = tf.cast(tf.shape(label)[0],'int64')*tf.ones([1], 'int64')
    #    #extra = tf.Print(extra, [extra], "extra", summarize=1000)
    #    cols = tf.concat([cols,extra],0)
    #    #cols = tf.Print(cols,[cols],"cols",summarize =  1000)
    #    cols_len = tf.shape(cols)[0]
    #    cols_right_shifted = tf.concat([[-1], cols[:cols_len-1]], 0)
    #    words_len_final = cols - cols_right_shifted - 1
    #    words_len_final = tf.cast(words_len_final,'float')
    #    num_nonzeros = tf.count_nonzero(words_len_final)
    #    #words_len_sum = tf.cond(tf.equal(tf.cast(num_nonzeros, tf.float32), tf.constant(0.0,dtype=tf.float32)), tf.constant(0.0,dtype=tf.float32),tf.divide(tf.cast(tf.reduce_sum(words_len_final),tf.float32), tf.cast(num_nonzeros, tf.float32)))
    #   # #if tf.equal(tf.cast(num_nonzeros, tf.float32), tf.constant(0.0,dtype=tf.float32)):
    #    #    words_len_sum = tf.constant(0.0,dtype=tf.float32)
    #    #else:
    #        #words_len_sum = tf.divide(tf.cast(tf.reduce_sum(words_len_final),tf.float32), tf.cast(num_nonzeros, tf.float32))
    #    words_len_sum = tf.cond(tf.equal(tf.cast(num_nonzeros, tf.float32), tf.constant(0.0,dtype=tf.float32)), true_fn, lambda: false_fn1(num_nonzeros, words_len_final))
    #    w.append(words_len_sum)
    #    words_len_av = tf.reduce_mean(words_len_final,0)
    #    word_av_list.append(words_len_av)
    #word_av_tf = tf.convert_to_tensor(word_av_list)
    ##word_av_tf= tf.Print(word_av_tf,[word_av_tf],"word_av_tf",summarize =  1000)
    #w = tf.convert_to_tensor(w)
    #w = tf.Print(w,[w],"w",summarize =  1000)
    #median = get_real_median(w)
    ##norm_word_len = tf.cond(tf.equal(tf.cast(median, tf.float32), tf.constant(0.0,dtype=tf.float32)), tf.constant(0.0,dtype=tf.float32), tf.divide(w,median))
    ##if tf.equal(tf.cast(median, tf.float32), tf.constant(0.0,dtype=tf.float32)):
    ##    norm_word_len = tf.constant(0.0,dtype=tf.float32)
    ##else:
    ##    norm_word_len = tf.divide(w,median)
    #norm_word_len = tf.cond(tf.equal(tf.cast(median, tf.float32), tf.constant(0.0,dtype=tf.float32)), true_fn, lambda: false_fn2(w,median))
    #norm_word_len = tf.Print(norm_word_len, [norm_word_len], "norm_word_len", summarize = 1000)
    current_step  = 0
    shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=200, num_classes=148)
    global_step = tf.Variable(0, trainable=False)
    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)
    
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=200*np.ones(32),ignore_longer_outputs_than_inputs=True))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,200*np.ones(32), merge_repeated=False)

    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    # Set tf summary
    tboard_save_path = 'model/CRNN/tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    model_save_dir = 'model/CRNN/model/shadownet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_'#{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    #train_epochs = config.cfg.TRAIN.EPOCHS

    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(train_epochs):
            
            _, c, seq_distance, preds, gt_labels, summary = sess.run(
                [optimizer, cost, sequence_dist, decoded, input_labels, merge_summary_op])
            indices = gt_labels.indices
            values = gt_labels.values
            dense_shape = gt_labels.dense_shape
            #logger.info(indices.shape) 
            #log
            preds = decoder.sparse_tensor_to_str(preds[0])
            gt_labels = decoder.sparse_tensor_to_str(gt_labels)
            current_step += 1
            accuracy = []

            for index, gt_label in enumerate(gt_labels):
                pred = preds[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            #
            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                    epoch + 1, c, seq_distance, accuracy))

            summary_writer.add_summary(summary=summary, global_step=global_step)
            if steps_per_checkpoint is None:  
                saver.save(sess=sess, save_path=model_save_path, global_step=global_step)
            else:
                if current_step%int(steps_per_checkpoint)==0:
                    saver.save(sess=sess, save_path=model_save_path, global_step=global_step)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if not ops.exists(os.getcwd()+"/model/CRNN/data/tfReal/"+args.filename):
        raise ValueError('{:s} doesn\'t exist'.format(args.filename))

    train_shadownet(args.filename,  args.train_epochs, args.weights_path, args.steps_per_checkpoint)
print('Done')
