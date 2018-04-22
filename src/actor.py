#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:55:34 2018

@author: guille
"""
import tensorflow as tf

class Actor:
    def __init__(self,sess,state_size,action_size,learning_rate,action_range,hidden_size,name,subspace_name,minibatch_size):
        self.hidden_size=hidden_size
        self.sess=sess
        self.action_range=action_range
        self.state_input_tensor=tf.placeholder(tf.float32, shape=(None, state_size),name="state_input_tensor")
        self.Q_wrt_a=tf.placeholder(tf.float32, shape=(None, action_size),name="Q_wrt_a")
        self.name=name
        self.output=self.createActor(self.state_input_tensor,action_size)
        self.weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=(subspace_name+"/"+name+"_network"))
        extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=(subspace_name+"/"+name+"_network"))
        self.combined_grads=tf.gradients(self.output,self.weights,-self.Q_wrt_a,name="a_wrt_weights_X_Q_wrt_a")
        factor=1/float(minibatch_size)
        for e in range(len(self.combined_grads)):
            self.combined_grads[e]=tf.scalar_mul(factor,self.combined_grads[e])
        grads=zip(self.combined_grads,self.weights)
        self.train=tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
    def createActor(self,state_input_tensor,action_size):
        with tf.variable_scope(self.name+"_network"):
	    batch_norm_input=tf.layers.batch_normalization(state_input_tensor,name="Batch_norm_on_input",reuse=False)
            h1=tf.layers.dense(batch_norm_input,self.hidden_size,activation=tf.nn.relu,name="hidden_layer_1",reuse=False)
            batch_norm1=tf.layers.batch_normalization(h1,name="Batch_norm_on_h1",reuse=False)
            h2=tf.layers.dense(batch_norm1,self.hidden_size,activation=tf.nn.relu,name="hidden_layer_2",reuse=False)
            batch_norm2=tf.layers.batch_normalization(h2,name="Batch_norm_on_h2",reuse=False)
            h3=tf.layers.dense(batch_norm2,self.hidden_size,activation=tf.nn.relu,name="hidden_layer_3",reuse=False)
            batch_norm3=tf.layers.batch_normalization(h3,name="Batch_norm_on_h3",reuse=False)
            output=tf.multiply(tf.layers.dense(batch_norm3,action_size,activation=tf.nn.tanh,name="output_layer",reuse=False),self.action_range)
        return output
    def predict(self,state):
        feed_dict={self.state_input_tensor:state}
        return self.sess.run(self.output,feed_dict)
    def trainModel(self,states,Q_wrt_a):
        feed_dict={self.state_input_tensor:states,self.Q_wrt_a:Q_wrt_a}
        self.sess.run(self.train,feed_dict)
        self.sess.run(extra_update_ops)
