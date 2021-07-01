import argparse
import h5py
from math import *
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
import json
import os, ast
import sys

np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#import gapnet_seg as MODEL
import gapnet as MODEL


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='[10,1,16,64,128,1,64,128,128,256,128,128]', help='DNN parameters[[k,H,A,F,F,F,H,A,F,C,F]]')
parser.add_argument('--model_path', default='../logs', help='Model checkpoint path')
parser.add_argument('--name', default='test', help='Output model name')
parser.add_argument('--num_point', type=int, default=14, help='Point Number [default: 80]')
parser.add_argument('--nfeat', type=int, default=13, help='Number of features [default: 17]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 4]')

FLAGS = parser.parse_args()
MODEL_PATH = os.path.join(FLAGS.model_path,FLAGS.name)
params = ast.literal_eval(FLAGS.params)




# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
NFEATURES = FLAGS.nfeat
NUM_CATEGORIES = FLAGS.ncat

def load_pb(pb_model):
    with tf.gfile.GFile(pb_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
  
def save_eval():
    with tf.Graph().as_default():
        pointclouds_pl,  labels_pl = MODEL.placeholder_inputs(1, NUM_POINT,NFEATURES)
        print (pointclouds_pl.name,labels_pl.name)
                        
        is_training_pl = tf.placeholder(tf.bool, shape=())
        pred = MODEL.get_model(pointclouds_pl, is_training=is_training_pl,params=params,num_class=NUM_CATEGORIES,scname='PL')
        pred = tf.nn.softmax(pred)
        saver = tf.train.Saver()
          
        print (' is_training: ', is_training_pl.name)    
        config = tf.ConfigProto()
        sess = tf.Session(config=config)

        saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        print('model restored')

        builder = tf.saved_model.builder.SavedModelBuilder("../pretrained/{}".format(FLAGS.name))
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()


        #print([node.name for node in sess.graph.as_graph_def().node])
        print('prediction name:',pred.name)


            

################################################          
    
def load_eval():
    '''
    To evaluate the ABCNet score we have to create a tensorflow session, give the relevant distributions, and run the frozen model.
    '''
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], '../pretrained/{}'.format(FLAGS.name))
        flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('{} FLOPS after freezing'.format(flops.total_float_ops))
        output = sess.graph.get_tensor_by_name('Softmax_2:0')
        mock_data = np.ones((1,NUM_POINT,NFEATURES),dtype=float)
        #mock_label = np.ones((1,NUM_POINT),dtype=float)
        mock_label = np.ones((1),dtype=float)



        feed_dict = {
            'Placeholder:0': mock_data,
            'Placeholder_1:0': mock_label,
            'Placeholder_2:0': False,
        }

        predictions = sess.run(output, feed_dict)
        #predictions store 1 value per particle. The shape is [1,100,2]. The probability as coming from the B decay is stored at position 2, for example, the probability for the first particle come from the B decay is stored at predictions[0,0,1], while for the second is stored at predictions[0,1,1] and so on.
        for pred in predictions: 
            print (pred)
        

if __name__=='__main__':
  save_eval()
  #load_eval()
