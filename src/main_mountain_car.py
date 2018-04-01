#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:26:09 2018

@author: guille
"""
import replayMemory
import actor
import critic
import tensorflow as tf
import numpy as np
import mountain_car_parameters
from stochastic.diffusion import OrnsteinUhlenbeckProcess

#Environment
env=mountain_car_parameters.env
ACTION_RANGE=mountain_car_parameters.ACTION_RANGE
STATE_SIZE=mountain_car_parameters.STATE_SIZE
ACTION_SIZE=mountain_car_parameters.ACTION_SIZE
#Memory
MINIBATCH_SIZE=mountain_car_parameters.MINIBATCH_SIZE
MEMORY_MAX_SIZE=mountain_car_parameters.MEMORY_MAX_SIZE
INDEX_STATE=mountain_car_parameters.INDEX_STATE
INDEX_REWARD=mountain_car_parameters.INDEX_REWARD
INDEX_DONE=mountain_car_parameters.INDEX_DONE
INDEX_LAST_STATE=mountain_car_parameters.INDEX_LAST_STATE
INDEX_ACTION=mountain_car_parameters.INDEX_ACTION
VAR_SIZE_DIC=mountain_car_parameters.VAR_SIZE_DIC
#Actor hyperparameters
ACTOR_LEARNING_RATE=mountain_car_parameters.ACTOR_LEARNING_RATE
HIDDEN_SIZE_ACTOR=mountain_car_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=mountain_car_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=mountain_car_parameters.ACTOR_SUBSPACE_NAME
ACTOR_TARGET_NAME=mountain_car_parameters.ACTOR_TARGET_NAME
ACTOR_TARGET_SUBSPACE_NAME=mountain_car_parameters.ACTOR_TARGET_SUBSPACE_NAME
#Critic hyperparameters
CRITIC_L2_WEIGHT_DECAY=mountain_car_parameters.CRITIC_L2_WEIGHT_DECAY
CRITIC_LEARNING_RATE=mountain_car_parameters.CRITIC_LEARNING_RATE
HIDDEN_SIZE_CRITIC=mountain_car_parameters.HIDDEN_SIZE_CRITIC
CRITIC_NAME=mountain_car_parameters.CRITIC_NAME
CRITIC_SUBSPACE_NAME=mountain_car_parameters.CRITIC_SUBSPACE_NAME
CRITIC_TARGET_NAME=mountain_car_parameters.CRITIC_TARGET_NAME
CRITIC_TARGET_SUBSPACE_NAME=mountain_car_parameters.CRITIC_TARGET_SUBSPACE_NAME
#Q function parameters:
DISCOUNT=mountain_car_parameters.DISCOUNT
#Algorithm parameters:
LEARNING_HAS_STARTED=mountain_car_parameters.LEARNING_HAS_STARTED 
NUM_EPISODES=mountain_car_parameters.NUM_EPISODES
EPOCHS_PER_EPISODE=mountain_car_parameters.EPOCHS_PER_EPISODE
WARMUP=mountain_car_parameters.WARMUP
TRAINING_ITERATIONS_PER_EPISODE=mountain_car_parameters.TRAINING_ITERATIONS_PER_EPISODE
TAU=mountain_car_parameters.TAU
NOISE_FACTOR=mountain_car_parameters.NOISE_FACTOR
NOISE_MOD=mountain_car_parameters.NOISE_MOD
OU_SPEED=mountain_car_parameters.OU_SPEED
OU_MEAN=mountain_car_parameters.OU_MEAN
OU_VOLATILITY=mountain_car_parameters.OU_VOLATILITY
#Program parameters:
LOGS_PATH=mountain_car_parameters.LOGS_PATH
SAVE_PATH=mountain_car_parameters.SAVE_PATH
VISUALIZE=mountain_car_parameters.VISUALIZE
EPISODE_CHECKPOINT=mountain_car_parameters.EPISODE_CHECKPOINT
VISUALIZATION_CHECKPOINT=mountain_car_parameters.VISUALIZATION_CHECKPOINT
VISUALIZATION_EPISODES=mountain_car_parameters.VISUALIZATION_EPISODES
VISUALIZATION_ITERATIONS=mountain_car_parameters.VISUALIZATION_ITERATIONS
#Initialization
tf.reset_default_graph()
sess=tf.Session()
replayMemory=replayMemory.replayMemory(MINIBATCH_SIZE,MEMORY_MAX_SIZE,VAR_SIZE_DIC)
with tf.variable_scope(ACTOR_SUBSPACE_NAME):
    my_actor=actor.Actor(sess,STATE_SIZE,ACTION_SIZE,ACTOR_LEARNING_RATE,ACTION_RANGE,HIDDEN_SIZE_ACTOR,ACTOR_NAME,ACTOR_SUBSPACE_NAME,MINIBATCH_SIZE)
with tf.variable_scope(ACTOR_TARGET_SUBSPACE_NAME):
    my_actor_target=actor.Actor(sess,STATE_SIZE,ACTION_SIZE,ACTOR_LEARNING_RATE,ACTION_RANGE,HIDDEN_SIZE_ACTOR,ACTOR_TARGET_NAME,ACTOR_TARGET_SUBSPACE_NAME,MINIBATCH_SIZE)
with tf.variable_scope(CRITIC_SUBSPACE_NAME):
    my_critic=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,CRITIC_LEARNING_RATE,HIDDEN_SIZE_CRITIC,CRITIC_NAME,CRITIC_SUBSPACE_NAME,CRITIC_L2_WEIGHT_DECAY)
with tf.variable_scope(CRITIC_TARGET_SUBSPACE_NAME):
    my_critic_target=critic.Critic(sess,STATE_SIZE,ACTION_SIZE,CRITIC_LEARNING_RATE,HIDDEN_SIZE_CRITIC,CRITIC_TARGET_NAME,CRITIC_TARGET_SUBSPACE_NAME,CRITIC_L2_WEIGHT_DECAY)
update_target_ops=[]
with tf.variable_scope("TARGET_UPDATE"):
    for i in range(len(my_actor.weights)):
        update_target_op=my_actor_target.weights[i].assign(TAU*my_actor.weights[i]+(1-TAU)*my_actor_target.weights[i])
        update_target_ops.append(update_target_op)
    for i in range(len(my_critic.weights)):
        update_target_op=my_critic_target.weights[i].assign(TAU*my_critic.weights[i]+(1-TAU)*my_critic_target.weights[i])
        update_target_ops.append(update_target_op)
OUP=OrnsteinUhlenbeckProcess(t=EPOCHS_PER_EPISODE,speed=OU_SPEED,mean=OU_MEAN,vol=OU_VOLATILITY)
#Tensorboard
writer=tf.summary.FileWriter(LOGS_PATH,sess.graph)
loss_summary=tf.placeholder('float',name='Critic_loss_value')
reward_summary=tf.placeholder('float',name='Reward_value')
loss_sum=tf.summary.scalar("Critic_loss", loss_summary)
re_sum=tf.summary.scalar("reward", reward_summary)
summaryMerged=tf.summary.merge_all()
saver = tf.train.Saver()
init_op=tf.global_variables_initializer()
tf.get_default_graph().finalize()
sess.run(init_op)
#Start algorithm
loss=0
acc_reward=0
epoch=0
for episode in range(NUM_EPISODES):
    exploration_noise=OUP.sample(EPOCHS_PER_EPISODE-1)
    noise_scale=(NOISE_FACTOR*NOISE_MOD**episode)*ACTION_RANGE*10
    exploration_noise*=noise_scale
    state=env.reset()
    last_state=0
    for it in range(EPOCHS_PER_EPISODE):
        #Select action
        action=exploration_noise[it]+my_actor.predict(np.reshape(state,(1,STATE_SIZE)))
        if action>ACTION_RANGE:
            action[0][0]=ACTION_RANGE
        if action<-ACTION_RANGE:
            action[0][0]=-ACTION_RANGE
        #Store transition
        state, reward, done, info = env.step(action)
        state=np.reshape(state,(1,STATE_SIZE))
        if it>1:
            replayMemory.add(state,reward,done,last_state,action)
        acc_reward+=reward
        last_state=state
        epoch+=1
        #Train
        if (len(replayMemory.memory)>MINIBATCH_SIZE) and epoch>WARMUP:
            if not LEARNING_HAS_STARTED:
                print "Warmup phase over, starting training..."
                LEARNING_HAS_STARTED=True
            for i in range(TRAINING_ITERATIONS_PER_EPISODE):
                #Sample minibatch
                minibatch=replayMemory.get_batch()
                S=replayMemory.get_from_minibatch(minibatch,INDEX_STATE)
                St0=replayMemory.get_from_minibatch(minibatch,INDEX_LAST_STATE)
                A=replayMemory.get_from_minibatch(minibatch,INDEX_ACTION)
                D=replayMemory.get_from_minibatch(minibatch,INDEX_DONE)
                R=replayMemory.get_from_minibatch(minibatch,INDEX_REWARD)       
                #Calculate target Q with target networks
                target_Q=R+DISCOUNT*my_critic_target.predict(S,my_actor_target.predict(S))
                for element in range(len(target_Q)):
					if D[element][0]==True:
						target_Q[element]=R[element]
                #Update critic
                loss+=my_critic.trainModel(St0,A,target_Q)
                #Update actor
                my_actor.trainModel(St0,my_critic.getGradients(St0,my_actor.predict(St0))[0])   
                #Update target networks
                sess.run(update_target_ops)
        if done or it==EPOCHS_PER_EPISODE-1:
            #Tensorboard
            if LEARNING_HAS_STARTED:
                #Record Reward
                mean_reward=float(acc_reward)/it
                sumOut=sess.run(re_sum,feed_dict={reward_summary:mean_reward})
                writer.add_summary(sumOut,episode)
                #Record loss:
                mean_loss=float(loss)/it
                summary_loss=sess.run(loss_sum,feed_dict={loss_summary:mean_loss})
                writer.add_summary(summary_loss,episode)
            acc_reward=0
            loss=0
            break
    #Examine algorithm:
    if episode%EPISODE_CHECKPOINT==0 and episode!=0:
        print "Episode",episode,"of",NUM_EPISODES
        if episode%VISUALIZATION_CHECKPOINT==0 and VISUALIZE and LEARNING_HAS_STARTED:
			for e in range(VISUALIZATION_EPISODES):
				state=env.reset()
				for i in range(VISUALIZATION_ITERATIONS):
					env.render()
					state, reward, done, info=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
				env.render(close=True)
saver.save(sess,SAVE_PATH)
print "Model saved in path: ",SAVE_PATH
