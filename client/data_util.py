#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import logging
import PIL.Image as Image
import numpy as np
from defs import LMAP
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename=__name__+'.log',format='%(levelname)s:%(message)s', level=logging.DEBUG)


def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self,n_channels,x_features,y_features,base_path,n_classes):
        self.n_channels = n_channels
        self.x_features = x_features
        self.y_features = y_features
        self.base_path = base_path
        self.n_classes = n_classes

    def vectorize(self, examples,augm=False,test=False):
            #print(len(examples))
            im_data = np.zeros((len(examples), self.n_channels, self.y_features, self.x_features), dtype=np.float32)
            #print(np.shape(im_data))
            labels = []
            for i,example in enumerate(examples):
                    #print example
                    im = self.read_png(example[0],test)
                    if test:
                        im = im.resize((1040,780))
                    im_array = np.array(im).astype(np.float32)
                    #print(np.shape(im_array))
                    if augm:
                            offset = random.randint(0, 90)
                            im_data[i, :, :, :] = im_array[:self.y_features, offset:offset+self.x_features] / 256.0
                    else:
                            for j in range(self.n_channels):
                                im_data[i, j, :, :] = im_array[:self.y_features, :,j] / 256.0
                    #print(np.shape(im_data))
                    labels.append(LMAP[int(example[1])])
            return im_data,labels

    @classmethod
    def build(cls, data):
            return cls(data.n_channels, data.x_features, data.y_features,data.png_folder,data.n_classes)


    def load_and_preprocess_data(self,examples):
            #print(examples[0])
            #logger.info("Loading  data...%d ",len(examples))
            
            # now process all the input data.
            augm = False
            inputs,labels = self.vectorize(examples,augm)
            
            #logger.info("Done reading %d images", len(examples))

            return inputs,labels
    
    def load_and_preprocess_test_data(self,examples):
            
            # now process all the input data.
            inputs,labels = self.vectorize(examples,False,True)
            
            return inputs,labels

    def read_png(self,image,test):
            if test:
                return Image.open(self.base_path+str(image))
            else:
                return Image.open(self.base_path+str(image)+'_r.jpg')

def getModelHelper(args):
        helper = ModelHelper.build(args)
        return helper


class Config:
        """Holdsmodelhyperparamsanddatainformation.

        Theconfigclassisusedtostorevarioushyperparametersanddataset
        informationparameters.ModelobjectsarepassedaConfig()objectat
        instantiation.
        """
        n_channels=3
        x_features=1040
        y_features=780
        n_classes=6
        dropout=0.5
        batch_size=8
        n_epochs=1
        png_folder='./data/'

def testModelHelper():

        config = Config()

        helper = getModelHelper(config)
        
        list_ = []

        with open('./valEqual.csv') as valFile:
                lines = valFile.readlines()
                for line in lines:
                        split = line.split(',')
                        list_.append((split[0],split[1]))

        #list_ = [(l,m) for l,m in zip(list1,list2) ] 

        in_data,labels  = helper.load_and_preprocess_data(list_)

        y = {k:i for k,i in zip(list_,labels)}
        #print in_data[100],labels[100],list_[2900]

        #assert np.shape(in_data)== (8,1,128,858),np.shape(labels)==(8,3,)
if __name__ == "__main__":
        testModelHelper()

