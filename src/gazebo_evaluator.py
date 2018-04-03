#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gazebo_parameters
import tensorflow as tf
import numpy as np
import actor

# Loaded parameters:
env=gazebo_parameters.env
STATE_SIZE=gazebo_parameters.STATE_SIZE
ACTION_SIZE=gazebo_parameters.ACTION_SIZE
ACTOR_LEARNING_RATE=gazebo_parameters.ACTOR_LEARNING_RATE
ACTION_RANGE=gazebo_parameters.ACTION_RANGE
HIDDEN_SIZE_ACTOR=gazebo_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=gazebo_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=gazebo_parameters.ACTOR_SUBSPACE_NAME
MINIBATCH_SIZE=gazebo_parameters.MINIBATCH_SIZE
SAVE_PATH=gazebo_parameters.SAVE_PATH

# Evaluator parameters:
EPISODES_TO_EVALUATE=50
EPOCHS_PER_EPISODE_TO_EVALUATE=500


# Program start
tf.reset_default_graph()
sess=tf.Session()

with tf.variable_scope(ACTOR_SUBSPACE_NAME):
    my_actor=actor.Actor(sess,STATE_SIZE,ACTION_SIZE,ACTOR_LEARNING_RATE,ACTION_RANGE,HIDDEN_SIZE_ACTOR,ACTOR_NAME,ACTOR_SUBSPACE_NAME,MINIBATCH_SIZE)

saver = tf.train.Saver()

# Restore variables from disk.
saver.restore(sess,SAVE_PATH)
for e in range(EPISODES_TO_EVALUATE):
	state=env.reset()
	for i in range(EPOCHS_PER_EPISODE_TO_EVALUATE):
		state, reward=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
