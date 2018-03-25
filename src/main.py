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
import tensorflow as tf
import numpy as np
from stochastic.diffusion import OrnsteinUhlenbeckProcess

#Environment
ENVIRONMENT='Pendulum-v0'
env=gym.make(ENVIRONMENT)
ACTION_RANGE=env.action_space.high[0]
STATE_SIZE=env.observation_space.shape[0]
ACTION_SIZE=env.action_space.shape[0]
#Memory
MINIBATCH_SIZE=1024
MEMORY_MAX_SIZE=int(1e5)
INDEX_STATE=0
INDEX_REWARD=1
INDEX_DONE=2
INDEX_LAST_STATE=3
INDEX_ACTION=4
VAR_SIZE_DIC={INDEX_STATE:STATE_SIZE,
              INDEX_REWARD:1,
              INDEX_DONE:1,
              INDEX_LAST_STATE:STATE_SIZE,
              INDEX_ACTION:ACTION_SIZE}
#Actor hyperparameters
ACTOR_LEARNING_RATE=1e-4
HIDDEN_SIZE_ACTOR=8
ACTOR_NAME="actor"
ACTOR_SUBSPACE_NAME="ACTOR_OPS"
ACTOR_TARGET_NAME="actor_target"
ACTOR_TARGET_SUBSPACE_NAME="TARGET_ACTOR_OPS"
#Critic hyperparameters
CRITIC_L2_WEIGHT_DECAY=1e-2
CRITIC_LEARNING_RATE=1e-3
HIDDEN_SIZE_CRITIC=8
CRITIC_NAME="critic"
CRITIC_SUBSPACE_NAME="CRITIC_OPS"
CRITIC_TARGET_NAME="critic_target"
CRITIC_TARGET_SUBSPACE_NAME="TARGET_CRITIC_OPS"
#Q function parameters:
DISCOUNT=0.99
#Algorithm parameters:
LEARNING_HAS_STARTED=False #Don't change this, it's a flag
NUM_EPISODES=10000
EPOCHS_PER_EPISODE=200
WARMUP=100000
TRAINING_ITERATIONS_PER_EPISODE=1
TAU=1e-2
NOISE_FACTOR=0.1
NOISE_MOD=0.99
OU_SPEED=0.15
OU_MEAN=0
OU_VOLATILITY=0.2
#Program parameters:
LOGS_PATH="../logs"
SAVE_PATH="/tmp/model.ckpt"
VISUALIZE=True
EPISODE_CHECKPOINT=10
VISUALIZATION_CHECKPOINT=100
VISUALIZATION_ITERATIONS=500
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
    for e in range(EPOCHS_PER_EPISODE):
        exploration_noise[e]-=1
    NOISE_FACTOR*=NOISE_MOD
    state=env.reset()
    last_state=0
    it=0
    while True: #Goes on for 200 iterations
        #Select action
        noise=exploration_noise[it]+exploration_noise[it]*NOISE_FACTOR
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
saver.save(sess,SAVE_PATH)
print "Model saved in path: ",SAVE_PATH
"""
while(True):
    state=env.reset()
    for i in range(VISUALIZATION_ITERATIONS*2):
        env.render()
        state, reward, done, info=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
    env.render(close=True)
"""
