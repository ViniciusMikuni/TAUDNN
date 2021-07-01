import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util
from gat_layers import attn_feature


def placeholder_inputs(batch_size, num_point, num_features):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    seg_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, seg_pl


def gap_block(k,n_heads,nn_idx,net,point_cloud,edge_size,bn_decay,weight_decay,is_training,scname):
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(net, edge_size[1], nn_idx, activation=tf.nn.relu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer{0}'.format(edge_size[0])+scname, k=k, i=i)
        attns.append(edge_feature)# This is the edge feature * att. coeff. activated by RELU, one per particle
        local_features.append(locals) #Those are the yij


    neighbors_features = tf.concat(attns, axis=-1)
    #net = tf.squeeze(net)
    #neighbors_features = tf.concat([tf.expand_dims(net, -2), neighbors_features], axis=-1)                                                                                     
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)


    locals_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    return neighbors_features, locals_transform, coefs


def get_model(point_cloud, is_training, num_class,params, 
                weight_decay=None, bn_decay=None,scname=''):
    ''' input: BxNxF
    output:BxNx(cats*segms)  '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -2)

  
    k = params[0]
    adj = tf_util.pairwise_distanceR(point_cloud[:,:,:3])
    n_heads = params[1]
    nn_idx = tf_util.knn(adj, k=k)

    
    net, locals_transform, coefs= gap_block(k,n_heads,nn_idx,point_cloud,point_cloud,('filter0',params[2]),bn_decay,weight_decay,is_training,scname)
    
    
    net = tf_util.conv2d(net, params[3], [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet00', bn_decay=bn_decay)
    net00 = net

    net = tf_util.conv2d(net, params[4], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet01'+scname, bn_decay=bn_decay)
    
    net01 = tf.reduce_max(net, axis=-2, keep_dims=True)

    #k=5
    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)    
    adj_conv = nn_idx
    n_heads = params[5]

    net, locals_transform1, coefs2= gap_block(k,n_heads,nn_idx,net,point_cloud,('filter1',params[6]),bn_decay,weight_decay,is_training,scname)

    net = tf_util.conv2d(net, params[7], [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet10', bn_decay=bn_decay)
    net10 = net

    net = tf_util.conv2d(net, params[8], [1, 1], padding='VALID', stride=[1, 1], activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='gapnet11'+scname, bn_decay=bn_decay)
    
    #net11 = net
    net11 = tf.reduce_max(net, axis=-2, keep_dims=True)
    
    net = tf.concat([
        net00,
        net01,
        net10,
        net11,
        locals_transform,
        locals_transform1
    ], axis=-1)


    net = tf_util.conv2d(net, params[9], [1, 1], padding='VALID', stride=[1, 1], 
                         activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, scope='agg'+scname, bn_decay=bn_decay)
    net_tot = net
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='avgpool'+scname)

    expand = tf.tile(net, [1, num_point, 1, 1])
    net = tf.concat(axis=3, values=[expand, 
                                    #net_tot,                                    
                                    net01,
                                    net11,
                                    #global_expand,
                                    # locals_transform,
                                    # locals_transform1                                    
                                ])
    net = tf_util.conv2d(net, params[10], [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    # net = tf_util.conv2d(net, params[11], [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
    #         bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    # net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    net = tf_util.conv2d(net, params[11], [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
            bn=True, is_training=is_training, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    net = tf_util.conv2d(net, num_class, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
            bn=False, scope='seg/conv5', weight_decay=weight_decay, is_dist=True)

    net = tf.reshape(net, [batch_size, num_point, num_class])

    return net

  



def get_loss(seg_pred, seg):
    loss_per_part = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg)
    per_instance_seg_loss = tf.reduce_mean(loss_per_part, axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)        
    return seg_loss
  

