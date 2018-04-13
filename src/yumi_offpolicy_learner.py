#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import replayMemory
import actor
import critic
import yumi_parameters
import tensorflow as tf
import numpy as np
import csv

#Environment
env=yumi_parameters.env
ACTION_RANGE=yumi_parameters.ACTION_RANGE
STATE_SIZE=yumi_parameters.STATE_SIZE
ACTION_SIZE=yumi_parameters.ACTION_SIZE
#Memory
MINIBATCH_SIZE=yumi_parameters.MINIBATCH_SIZE
MEMORY_MAX_SIZE=yumi_parameters.MEMORY_MAX_SIZE
INDEX_STATE=yumi_parameters.INDEX_STATE
INDEX_REWARD=yumi_parameters.INDEX_REWARD
INDEX_DONE=yumi_parameters.INDEX_DONE
INDEX_LAST_STATE=yumi_parameters.INDEX_LAST_STATE
INDEX_ACTION=yumi_parameters.INDEX_ACTION
VAR_SIZE_DIC=yumi_parameters.VAR_SIZE_DIC
#Actor hyperparameters
ACTOR_LEARNING_RATE=yumi_parameters.ACTOR_LEARNING_RATE
HIDDEN_SIZE_ACTOR=yumi_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=yumi_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=yumi_parameters.ACTOR_SUBSPACE_NAME
ACTOR_TARGET_NAME=yumi_parameters.ACTOR_TARGET_NAME
ACTOR_TARGET_SUBSPACE_NAME=yumi_parameters.ACTOR_TARGET_SUBSPACE_NAME
#Critic hyperparameters
CRITIC_L2_WEIGHT_DECAY=yumi_parameters.CRITIC_L2_WEIGHT_DECAY
CRITIC_LEARNING_RATE=yumi_parameters.CRITIC_LEARNING_RATE
HIDDEN_SIZE_CRITIC=yumi_parameters.HIDDEN_SIZE_CRITIC
CRITIC_NAME=yumi_parameters.CRITIC_NAME
CRITIC_SUBSPACE_NAME=yumi_parameters.CRITIC_SUBSPACE_NAME
CRITIC_TARGET_NAME=yumi_parameters.CRITIC_TARGET_NAME
CRITIC_TARGET_SUBSPACE_NAME=yumi_parameters.CRITIC_TARGET_SUBSPACE_NAME
#Q function parameters:
DISCOUNT=yumi_parameters.DISCOUNT
#Algorithm parameters:
LEARNING_HAS_STARTED=yumi_parameters.LEARNING_HAS_STARTED
NUM_EPISODES=yumi_parameters.NUM_EPISODES
EPOCHS_PER_EPISODE=yumi_parameters.EPOCHS_PER_EPISODE
WARMUP=yumi_parameters.WARMUP
EPISODES_PER_RECORD=yumi_parameters.EPISODES_PER_RECORD
TRAINING_ITERATIONS_PER_EPISODE=yumi_parameters.TRAINING_ITERATIONS_PER_EPISODE
TAU=yumi_parameters.TAU
NOISE_FACTOR=yumi_parameters.NOISE_FACTOR
NOISE_MOD=yumi_parameters.NOISE_MOD
OU_SPEED=yumi_parameters.OU_SPEED
OU_MEAN=yumi_parameters.OU_MEAN
OU_VOLATILITY=yumi_parameters.OU_VOLATILITY
#Program parameters:
LOGS_PATH=yumi_parameters.LOGS_PATH
SAVE_PATH=yumi_parameters.SAVE_PATH
RESTORE_PREVIOUS_SESSION=yumi_parameters.RESTORE_PREVIOUS_SESSION
VISUALIZE=yumi_parameters.VISUALIZE
EPISODE_CHECKPOINT=yumi_parameters.EPISODE_CHECKPOINT
#Initialization
tf.reset_default_graph()
sess=tf.Session()
replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)
with tf.variable_scope(ACTOR_SUBSPACE_NAME):
    my_actor=actor.Actor(sess,STATE_SIZE,ACTION_SIZE,ACTOR_LEARNING_RATE,ACTION_RANGE,HIDDEN_SIZE_ACTOR,ACTOR_NAME,ACTOR_SUBSPACE_NAME,MINIBATCH_SIZE)
with tf.variable_scope(CRITIC_SUBSPACE_NAME):
    my_critic=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,CRITIC_LEARNING_RATE,HIDDEN_SIZE_CRITIC,CRITIC_NAME,CRITIC_SUBSPACE_NAME,CRITIC_L2_WEIGHT_DECAY)

saver = tf.train.Saver()

#Restore previous session?
if RESTORE_PREVIOUS_SESSION:
	saver.restore(sess,SAVE_PATH)
	print "Restored previous session"
	tf.get_default_graph().finalize()
else:
	init_op=tf.global_variables_initializer()
	tf.get_default_graph().finalize()
	sess.run(init_op)

f = open("yumi_generated_data.csv","r")
cr = csv.reader(f,delimiter=';')
for row in cr:
	state=row[0]
	action=row[1]
	reward=row[2]
	replayMemory.add(0,reward,True,state,action)

#Train
for episode in range(100):
	minibatch=replayMemory.get_batch()
	St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
	A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
	R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)
	#Calculate target Q with target networks
	target_Q=R
	#Update critic
	loss+=my_critic.trainModel(St0,A,target_Q)
	#Update actor
	my_actor.trainModel(St0,my_critic.getGradients(St0,my_actor.predict(St0))[0]) 

saver.save(sess,SAVE_PATH)
print "Model saved in path: ",SAVE_PATH
