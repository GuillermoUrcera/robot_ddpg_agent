#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:38:28 2018

@author: guille
"""
import tensorflow as tf

class Critic:
    def __init__(self,sess,state_size,action_size,learning_rate,hidden_size,name,subspace_name,L2):
        self.sess=sess
        self.learning_rate=learning_rate
        self.hidden_size=hidden_size
        self.name=name
        self.state_input_tensor=tf.placeholder(tf.float32, shape=(None, state_size),name="state_input_tensor")
        self.action_input_tensor=tf.placeholder(tf.float32, shape=(None, action_size),name="action_input_tensor")
        self.output=self.createCritic(state_size,action_size)
        self.target_Q=tf.placeholder(tf.float32,shape=(None,1),name="target_Q")
        self.loss=tf.reduce_mean(tf.square(self.target_Q-self.output),name="loss")
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=(subspace_name+"/"+name+"_network"))
        for weight in self.weights:
			if not 'bias' in weight.name:
				self.loss+=L2*tf.nn.l2_loss(weight)
        self.Q_wrt_a_grads_op=tf.gradients(self.output,self.action_input_tensor,name="Q_wrt_a")
        self.train=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,var_list=self.weights)
    def createCritic(self,state_size,action_size):
        with tf.variable_scope(self.name+"_network"):
            critic_input=tf.layers.dense(self.state_input_tensor,self.hidden_size,activation=tf.nn.relu,name="input_layer_1",reuse=False)
            concat_layer=tf.concat([critic_input,self.action_input_tensor],1)
            h1=tf.layers.dense(concat_layer,self.hidden_size,activation=tf.nn.relu,name="hidden_layer_1",reuse=False)
            h2=tf.layers.dense(h1,self.hidden_size,activation=tf.nn.relu,name="hidden_layer_2",reuse=False)
            h3=tf.layers.dense(h2,self.hidden_size,activation=tf.nn.relu,name="hidden_layer_3",reuse=False)
            output=tf.layers.dense(h3,1,activation=None,name="output_layer")#None is linear
        return output
    def predict(self,state,action):
        feed_dict={self.state_input_tensor:state,self.action_input_tensor:action}
        return self.sess.run(self.output,feed_dict)
    def trainModel(self,state,action,target_Q):
        feed_dict={self.state_input_tensor:state,self.action_input_tensor:action,self.target_Q:target_Q}
        self.sess.run(self.train,feed_dict)
        return self.sess.run(self.loss,feed_dict)
    def getLoss(self,state,action,target_Q):
        feed_dict={self.state_input_tensor:state,self.action_input_tensor:action,self.target_Q:target_Q}
        return self.sess.run(self.loss,feed_dict)
    def getGradients(self,state,action):
        feed_dict={self.action_input_tensor:action,self.state_input_tensor:state}
        return self.sess.run(self.Q_wrt_a_grads_op,feed_dict)
        
        
