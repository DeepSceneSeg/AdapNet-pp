''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.'''



import tensorflow as tf
from dataset.helper import *
import numpy as np
import argparse
import os
import datetime
import importlib
import yaml
import re
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config')

def test_func(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.'+config['model'])
    model_func = getattr(module,config['model'])     
    data_list, iterator = get_test_data(config)
    resnet_name='resnet_v1_50'
    with tf.variable_scope(resnet_name):
    	model = model_func(num_classes=config['num_classes'],training=False)
        images_pl = tf.placeholder(tf.float32, [None, config['height'],config['width'], 3])
        logits = model.build_graph(images_pl)
    config1 =  tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess=tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print 'total_variables_loaded:',len(import_variables)  
   
    saver = tf.train.Saver(import_variables)
    saver.restore(sess, config['checkpoint'])
    sess.run(iterator.initializer)
    step=0
    total_num=0
    output_matrix=np.zeros([config['num_classes'],3])
    while 1:
    	try:
            img,label = sess.run([data_list[0],data_list[1]])
           
            feed_dict={images_pl : img }
            probabilities = sess.run([model.softmax],feed_dict=feed_dict)
            prediction=np.argmax(probabilities[0],3)
            gt=np.argmax(label,3)
            prediction[gt==0]=0
            output_matrix=compute_output_matrix(gt,prediction,output_matrix)
            total_num += label.shape[0]
            if (step+1) % config['skip_step'] == 0:
               print '%s %s] %d. iou updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), total_num)
               print 'mIoU: ',compute_iou(output_matrix)
            #compute_iou(confusion_matrix)
            step += 1
            #print 'total_data: ',total_num
        except tf.errors.OutOfRangeError: 
            print 'mIoU: ',compute_iou(output_matrix),'total_data: ',total_num
            break



def main():
    args = parser.parse_args()
    if args.config:
        f=open(args.config)
        config=yaml.load(f)
    else:
        print '--config config_file_address missing'
    test_func(config)

if __name__=='__main__':
        main()
