#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A combination of COVNET AND RNN nets for Lang Detection
@Author: Sahil Badyal
"""

import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from models import  data_util
from models.defs import LBLS
from models.data_util import getModelHelper
from models.defect_detection_model import DefectDetectionModel

logger = logging.getLogger("ld.cov.rnn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_channels = 3
    y_features = 780
    x_features = 1040
    n_classes = 6
    batch_size = 4
    n_epochs = 40
    lr = 0.001
    png_folder = ''

    def __init__(self, args):

            #if "model_path" in args.keys():
                    # Where to save things.
            self.output_path = './'
            #else:
            #        self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
            #self.model_output = self.output_path + "model.weights"
            #self.eval_output = self.output_path + "results.txt"
            self.log_output = self.output_path + "log"
            self.png_folder = args['png_folder']

class COVNETModel(DefectDetectionModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        self.input_placeholder = tf.placeholder(shape=[None,self.config.n_channels,self.config.y_features,self.config.x_features],dtype=tf.float32)
        self.labels_placeholder = tf.placeholder(shape=[None,self.config.n_classes],dtype=tf.int32)
        self.dropout_placeholder = tf.placeholder(shape=(),dtype=tf.float32)
        self.isTraining = tf.placeholder(shape=(),dtype=tf.bool)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0,training=False):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
                self.input_placeholder:inputs_batch,
                self.dropout_placeholder:  dropout,
                self.isTraining:training
                }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict
    
    def add_prediction_op(self):
        """Adds the following network:

        Returns:
            pred: tf.Tensor of shape (batch_size,)
        """
        input_layer  = self.input_placeholder

        dropout_rate = self.dropout_placeholder


        input_layer = tf.reshape(input_layer,shape=[-1,self.config.y_features,self.config.x_features,self.config.n_channels])


        conv1 = tf.layers.conv2d(
                      inputs=input_layer,
                      filters=16,
                      kernel_size=[3, 3],
                      padding="same",
                      activation=tf.nn.relu)

        output_cnn_1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[7, 7], strides=5,padding='same')
        
        
        self.shapeOfCNN1 = tf.shape(output_cnn_1)


        #conv2 = tf.layers.conv2d(
        #              inputs=output_cnn_1,
        #              filters=32,
        #              kernel_size=[3, 3],
        #              padding="same",
        #              activation=tf.nn.relu)
        #output_cnn = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=2,padding='same')
        
        #self.shapeOfCNN2 = tf.shape(output_cnn)

        
        
        filter_H =  156
        filter_W =  208
        num_channels = 16

        
        outputs = tf.reshape(output_cnn_1,shape=[-1,num_channels*filter_H*filter_W])


        ##Add fully connected layer
        with tf.name_scope('affine_layer'):
                preds = tf.contrib.layers.fully_connected(outputs,self.config.n_classes, activation_fn=None)
        tf.summary.histogram('preds',preds)

        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_placeholder,logits=preds))
        tf.summary.scalar('loss',loss)
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = None
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        if False:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                        with tf.control_dependencies(update_ops):
                                train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        else:
                train_op = optimizer.minimize(loss)
        return train_op

    def preprocess_speech_data(self, examples):
        """
        This function accepts examples and uses model helper to convert them into vectors 
        Args:
            examples: a list of examples of format (input,label)
        Returns:
            inputs,labels: The vectorized input and labels for training.

        """
        return self.helper.load_and_preprocess_data(examples)
    
    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        #print preds
        for i, (image, label) in enumerate(examples_raw):
            labels_ = preds[i] # only select elements of mask.
            ret.append([image, int(label), labels_])
        return ret
    
    def predict_on_batch(self, sess, inputs_batch,labels_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,labels_batch=labels_batch,training=False)
        predictions,loss,_summary = sess.run([tf.argmax(self.pred, axis=1),self.loss,self.summary], feed_dict=feed)
        return predictions,loss,_summary

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,dropout=0,training=True)
        #shapeCNN1, shapeCNN2 = sess.run([self.shapeOfCNN1,self.shapeOfCNN2], feed_dict=feed)
        #print(shapeCNN1, shapeCNN2)
        _, loss,_summary = sess.run([self.train_op, self.loss,self.summary], feed_dict=feed)
        return loss,_summary

    def __init__(self, helper, config):
        super(COVNETModel, self).__init__(helper, config)

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.isTraining = None
        self.shapeOfCNN1 = None
        self.shapeOfCNN2 = None
        #self.shapeOfCNN3 = None
        #self.shapeOfCNN4 = None
        

        self.build()


def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper_args = {
                'n_channels':config.n_channels,
                'x_features':config.x_features,
                'y_features':config.y_features,
                'base_path' :config.png_folder,
                'n_classes':config.n_classes
    }
    helper = getModelHelper(config)

    train = args['train_list']
    dev = args['val_list']

    #helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = COVNETModel(helper, config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.setSummaryWriters(session)
            model.fit(session, saver, train, dev)
            #if report:
            #    report.log_output(model.output(session, dev))
            #    report.save()
            #else:
                # Save predictions in a text file.
            #    output = model.output(session, dev)
            #    images, labels, predictions = zip(*output)
            #    predictions = [[LBLS[l] for l in preds] for preds in predictions]
            #    output = zip(images, labels, predictions)

            #    with open(model.config.conll_output, 'w') as f:
            #        write_csv(f, output)
                #with open(model.config.eval_output, 'w') as f:
                #    for sentence, labels, predictions in output:
                #        print_sentence(f, sentence, labels, predictions)

def name():
        return "2LCOVNETModel"

def do_evaluate(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = COVNETModel(helper, config)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)
