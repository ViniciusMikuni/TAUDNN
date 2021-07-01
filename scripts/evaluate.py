import argparse
import h5py
from math import *
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os, ast
import sys
from sklearn import metrics

#np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#from MVA_cfg import *
import provider
import gapnet_seg as MODEL
#import  gapnet_classify_global as model


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='[10,1,16,64,128,1,64,128,128,256,128,128]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='../logs/TAU', help='Model checkpoint path')
parser.add_argument('--batch', type=int, default=64, help='Batch Size  during training [default: 64]')
parser.add_argument('--num_point', type=int, default=80, help='Point Number [default: 500]')
parser.add_argument('--data_dir', default='../h5', help='directory with data [default: ../data/PU]')
parser.add_argument('--nfeat', type=int, default=15, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=5, help='Number of categories [default: 2]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')

FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
params = ast.literal_eval(FLAGS.params)
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  

# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat


NUM_CATEGORIES = FLAGS.ncat
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))



    
print('### Starting evaluation')


EVALUATE_FILE = os.path.join(H5_DIR, 'eval.h5')
  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl,  labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES)
          
            batch = tf.Variable(0, trainable=False)
                        
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,params=params,num_class=NUM_CATEGORIES,scname='PL')                        

            pred=tf.nn.softmax(pred)
            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,        
               'is_training_pl': is_training_pl,
               'pred': pred,}
            
        eval_one_epoch(sess,ops)

def get_batch(data,label, start_idx, end_idx):
    batch_label = label[start_idx:end_idx,:]
    batch_data = data[start_idx:end_idx,:,:]
    return batch_data, batch_label

        
def eval_one_epoch(sess,ops):
    is_training = False
    y_pred = []


    current_file = os.path.join(H5_DIR,EVALUATE_FILE)
    current_data, current_label = provider.load_h5(current_file,'seg')

    current_label = np.squeeze(current_label)
        
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE        
    #num_batches = 1
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label = get_batch(current_data, current_label, start_idx, end_idx)
            
        cur_batch_size = end_idx-start_idx


        feed_dict = {ops['pointclouds_pl']: batch_data,                  
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,
                 }

        
        pred = sess.run(ops['pred'],feed_dict=feed_dict)         
        if len(y_pred)==0:
            y_pred= pred
            y_doca = batch_data[:,:,5]
            y_lab = batch_label
        else:
            y_pred=np.concatenate((y_pred,pred),axis=0)
            y_doca=np.concatenate((y_doca,batch_data[:,:,5]),axis=0)
            y_lab=np.concatenate((y_lab,batch_label),axis=0)

    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:
        dset = fh5.create_dataset("DNN", data=y_pred)
        dset = fh5.create_dataset("doca", data=y_doca)
        dset = fh5.create_dataset("pid", data=y_lab)


################################################          
    

if __name__=='__main__':
  eval()
