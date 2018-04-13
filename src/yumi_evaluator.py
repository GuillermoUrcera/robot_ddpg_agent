#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import yumi_parameters
import tensorflow as tf
import numpy as np
import actor

# Loaded parameters:
env=yumi_parameters.env
STATE_SIZE=yumi_parameters.STATE_SIZE
ACTION_SIZE=yumi_parameters.ACTION_SIZE
ACTOR_LEARNING_RATE=yumi_parameters.ACTOR_LEARNING_RATE
ACTION_RANGE=yumi_parameters.ACTION_RANGE
HIDDEN_SIZE_ACTOR=yumi_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=yumi_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=yumi_parameters.ACTOR_SUBSPACE_NAME
MINIBATCH_SIZE=yumi_parameters.MINIBATCH_SIZE
SAVE_PATH=yumi_parameters.SAVE_PATH

# Evaluator parameters:
EPISODES_TO_EVALUATE=50

# Program start
tf.reset_default_graph()
sess=tf.Session()

with tf.variable_scope(ACTOR_SUBSPACE_NAME):
    my_actor=actor.Actor(sess,STATE_SIZE,ACTION_SIZE,ACTOR_LEARNING_RATE,ACTION_RANGE,HIDDEN_SIZE_ACTOR,ACTOR_NAME,ACTOR_SUBSPACE_NAME,MINIBATCH_SIZE)

saver = tf.train.Saver()

# Restore variables from disk.
saver.restore(sess,SAVE_PATH)
for e in range(EPISODES_TO_EVALUATE):
	raw_input("Press Enter to continue...")
	state=env.getState()
	reward=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
