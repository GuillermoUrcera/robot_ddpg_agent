#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:26:09 2018

@author: guille
"""
import gym
import replayMemory
import actor
import critic
import pendulum_parameters
import tensorflow as tf
import numpy as np
from stochastic.diffusion import OrnsteinUhlenbeckProcess

#Environment
env=pendulum_parameters.env
ACTION_RANGE=pendulum_parameters.ACTION_RANGE
STATE_SIZE=pendulum_parameters.STATE_SIZE
ACTION_SIZE=pendulum_parameters.ACTION_SIZE
#Memory
MINIBATCH_SIZE=pendulum_parameters.MINIBATCH_SIZE
MEMORY_MAX_SIZE=pendulum_parameters.MEMORY_MAX_SIZE
INDEX_STATE=pendulum_parameters.INDEX_STATE
INDEX_REWARD=pendulum_parameters.INDEX_REWARD
INDEX_DONE=pendulum_parameters.INDEX_DONE
INDEX_LAST_STATE=pendulum_parameters.INDEX_LAST_STATE
INDEX_ACTION=pendulum_parameters.INDEX_ACTION
VAR_SIZE_DIC=pendulum_parameters.VAR_SIZE_DIC
#Actor hyperparameters
ACTOR_LEARNING_RATE=pendulum_parameters.ACTOR_LEARNING_RATE
HIDDEN_SIZE_ACTOR=pendulum_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=pendulum_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=pendulum_parameters.ACTOR_SUBSPACE_NAME
ACTOR_TARGET_NAME=pendulum_parameters.ACTOR_TARGET_NAME
ACTOR_TARGET_SUBSPACE_NAME=pendulum_parameters.ACTOR_TARGET_SUBSPACE_NAME
#Critic hyperparameters
CRITIC_L2_WEIGHT_DECAY=pendulum_parameters.CRITIC_L2_WEIGHT_DECAY
CRITIC_LEARNING_RATE=pendulum_parameters.CRITIC_LEARNING_RATE
HIDDEN_SIZE_CRITIC=pendulum_parameters.HIDDEN_SIZE_CRITIC
CRITIC_NAME=pendulum_parameters.CRITIC_NAME
CRITIC_SUBSPACE_NAME=pendulum_parameters.CRITIC_SUBSPACE_NAME
CRITIC_TARGET_NAME=pendulum_parameters.CRITIC_TARGET_NAME
CRITIC_TARGET_SUBSPACE_NAME=pendulum_parameters.CRITIC_TARGET_SUBSPACE_NAME
#Q function parameters:
DISCOUNT=pendulum_parameters.DISCOUNT
#Algorithm parameters:
LEARNING_HAS_STARTED=pendulum_parameters.LEARNING_HAS_STARTED
NUM_EPISODES=pendulum_parameters.NUM_EPISODES
EPOCHS_PER_EPISODE=pendulum_parameters.EPOCHS_PER_EPISODE
WARMUP=pendulum_parameters.WARMUP
TRAINING_ITERATIONS_PER_EPISODE=pendulum_parameters.TRAINING_ITERATIONS_PER_EPISODE
TAU=pendulum_parameters.TAU
NOISE_FACTOR=pendulum_parameters.NOISE_FACTOR
NOISE_MOD=pendulum_parameters.NOISE_MOD
OU_SPEED=pendulum_parameters.OU_SPEED
OU_MEAN=pendulum_parameters.OU_MEAN
OU_VOLATILITY=pendulum_parameters.OU_VOLATILITY
#Program parameters:
LOGS_PATH=pendulum_parameters.LOGS_PATH
SAVE_PATH=pendulum_parameters.SAVE_PATH
VISUALIZE=pendulum_parameters.VISUALIZE
EPISODE_CHECKPOINT=pendulum_parameters.EPISODE_CHECKPOINT
VISUALIZATION_CHECKPOINT=pendulum_parameters.VISUALIZATION_CHECKPOINT
VISUALIZATION_ITERATIONS=pendulum_parameters.VISUALIZATION_ITERATIONS
ONLINE_EVALUATION=pendulum_parameters.ONLINE_EVALUATION
ONLINE_EVALUATION_EPISODES=pendulum_parameters.ONLINE_EVALUATION_EPISODES
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
Q_clean_summary=tf.placeholder('float',name='Clean_Q_value')
reward_clean_summary=tf.placeholder('float',name='Clean_reward_value')
loss_clean_summary=tf.placeholder('float',name='Loss_reward_value')
clean_q_sum=tf.summary.scalar("Q", Q_clean_summary)
clean_reward_sum=tf.summary.scalar("reward", reward_clean_summary)
clean_loss_sum=tf.summary.scalar("Critic loss", loss_clean_summary)
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
    noise_scale=(NOISE_FACTOR*NOISE_MOD**episode)*ACTION_RANGE
    exploration_noise*=noise_scale
    state=env.reset()
    last_state=0
    it=0
    while True: #Goes on for 200 iterations
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
        it+=1
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
                #Update critic
                loss+=my_critic.trainModel(St0,A,target_Q)
                #Update actor
                my_actor.trainModel(St0,my_critic.getGradients(St0,my_actor.predict(St0))[0])   
                #Update target networks
                sess.run(update_target_ops)
        if done:
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
                        for e in range(2):
				state=env.reset()
				for i in range(VISUALIZATION_ITERATIONS):
					env.render()
					state, reward, done, info=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
				env.render(close=True)
    if LEARNING_HAS_STARTED and ONLINE_EVALUATION:
        #Evaluate Q values, critic loss and reward
        reward_clean_summary=0
        Q_clean_summary=0
        loss_clean_summary=0
        state=env.reset()
        last_state=0
        for episode in range(ONLINE_EVALUATION_EPISODES):
            while True: #Goes on for 200 iterations
                state=np.reshape(state,(1,STATE_SIZE))
                action=my_actor.predict(state)
                if action>ACTION_RANGE:
                    action[0][0]=ACTION_RANGE
                if action<-ACTION_RANGE:
                    action[0][0]=-ACTION_RANGE
                last_state=state
                Q_clean_summary+=my_critic.predict(state,action)
                state, reward, done, info = env.step(action)
                reward_clean_summary+=reward
                target_Q=reward+DISCOUNT*my_critic_target.predict(state,my_actor_target.predict(state))
                loss_clean_summary+=my_critic.getLoss(last_state,action,target_Q)
                if done:
                    break
        writer.add_summary(sess.run(clean_q_sum,feed_dict={Q_clean_summary:tf.div(Q_clean_summary,ONLINE_EVALUATION_EPISODES*EPOCHS_PER_EPISODE)}),episode)
        writer.add_summary(sess.run(clean_reward_sum,feed_dict={reward_clean_summary:tf.div(reward_clean_summary,ONLINE_EVALUATION_EPISODES*EPOCHS_PER_EPISODE)}),episode)
        writer.add_summary(sess.run(clean_loss_sum,feed_dict={loss_clean_summary:tf.div(loss_clean_summary,ONLINE_EVALUATION_EPISODES*EPOCHS_PER_EPISODE)}),episode)
saver.save(sess,SAVE_PATH)
print "Model saved in path: ",SAVE_PATH
