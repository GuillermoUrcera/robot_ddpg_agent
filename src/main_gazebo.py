#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import replayMemory
import actor
import critic
import gazebo_parameters
import tensorflow as tf
import numpy as np
from stochastic.diffusion import OrnsteinUhlenbeckProcess

#Environment
env=gazebo_parameters.env
ACTION_RANGE=gazebo_parameters.ACTION_RANGE
STATE_SIZE=gazebo_parameters.STATE_SIZE
ACTION_SIZE=gazebo_parameters.ACTION_SIZE
#Memory
MINIBATCH_SIZE=gazebo_parameters.MINIBATCH_SIZE
MEMORY_MAX_SIZE=gazebo_parameters.MEMORY_MAX_SIZE
INDEX_STATE=gazebo_parameters.INDEX_STATE
INDEX_REWARD=gazebo_parameters.INDEX_REWARD
INDEX_DONE=gazebo_parameters.INDEX_DONE
INDEX_LAST_STATE=gazebo_parameters.INDEX_LAST_STATE
INDEX_ACTION=gazebo_parameters.INDEX_ACTION
VAR_SIZE_DIC=gazebo_parameters.VAR_SIZE_DIC
#Actor hyperparameters
ACTOR_LEARNING_RATE=gazebo_parameters.ACTOR_LEARNING_RATE
HIDDEN_SIZE_ACTOR=gazebo_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=gazebo_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=gazebo_parameters.ACTOR_SUBSPACE_NAME
ACTOR_TARGET_NAME=gazebo_parameters.ACTOR_TARGET_NAME
ACTOR_TARGET_SUBSPACE_NAME=gazebo_parameters.ACTOR_TARGET_SUBSPACE_NAME
#Critic hyperparameters
CRITIC_L2_WEIGHT_DECAY=gazebo_parameters.CRITIC_L2_WEIGHT_DECAY
CRITIC_LEARNING_RATE=gazebo_parameters.CRITIC_LEARNING_RATE
HIDDEN_SIZE_CRITIC=gazebo_parameters.HIDDEN_SIZE_CRITIC
CRITIC_NAME=gazebo_parameters.CRITIC_NAME
CRITIC_SUBSPACE_NAME=gazebo_parameters.CRITIC_SUBSPACE_NAME
CRITIC_TARGET_NAME=gazebo_parameters.CRITIC_TARGET_NAME
CRITIC_TARGET_SUBSPACE_NAME=gazebo_parameters.CRITIC_TARGET_SUBSPACE_NAME
#Q function parameters:
DISCOUNT=gazebo_parameters.DISCOUNT
#Algorithm parameters:
LEARNING_HAS_STARTED=gazebo_parameters.LEARNING_HAS_STARTED
NUM_EPISODES=gazebo_parameters.NUM_EPISODES
EPOCHS_PER_EPISODE=gazebo_parameters.EPOCHS_PER_EPISODE
WARMUP=gazebo_parameters.WARMUP
EPISODES_PER_RECORD=gazebo_parameters.EPISODES_PER_RECORD
TRAINING_ITERATIONS_PER_EPISODE=gazebo_parameters.TRAINING_ITERATIONS_PER_EPISODE
TAU=gazebo_parameters.TAU
NOISE_FACTOR=gazebo_parameters.NOISE_FACTOR
NOISE_MOD=gazebo_parameters.NOISE_MOD
OU_SPEED=gazebo_parameters.OU_SPEED
OU_MEAN=gazebo_parameters.OU_MEAN
OU_VOLATILITY=gazebo_parameters.OU_VOLATILITY
#Program parameters:
LOGS_PATH=gazebo_parameters.LOGS_PATH
SAVE_PATH=gazebo_parameters.SAVE_PATH
VISUALIZE=gazebo_parameters.VISUALIZE
EPISODE_CHECKPOINT=gazebo_parameters.EPISODE_CHECKPOINT
VISUALIZATION_CHECKPOINT=gazebo_parameters.VISUALIZATION_CHECKPOINT
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
OUP=OrnsteinUhlenbeckProcess(t=env.NUM_VIAPOINTS,speed=OU_SPEED,mean=OU_MEAN,vol=OU_VOLATILITY)
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
done=True
for episode in range(NUM_EPISODES):
    exploration_noise=OUP.sample(ACTION_SIZE)
    exploration_noise=exploration_noise[1:]
    noise_scale=(NOISE_FACTOR*NOISE_MOD**episode)*ACTION_RANGE
    exploration_noise*=noise_scale
    state=env.reset()
    last_state=state
    for it in range(EPOCHS_PER_EPISODE):
        #Select action
        action=exploration_noise+my_actor.predict(np.reshape(state,(1,STATE_SIZE)))
        for vp in range(ACTION_SIZE):
			if action[0][vp]>ACTION_RANGE:
				action[0][vp]=ACTION_RANGE
			if action[0][vp]<-ACTION_RANGE:
				action[0][vp]=-ACTION_RANGE
        #Store transition
        state, reward= env.step(action)
        state=np.reshape(state,(1,STATE_SIZE))
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
                target_Q=R
                #Update critic
                loss+=my_critic.trainModel(St0,A,target_Q)
                #Update actor
                my_actor.trainModel(St0,my_critic.getGradients(St0,my_actor.predict(St0))[0])   
                #Update target networks
                sess.run(update_target_ops)
        if episode%EPISODES_PER_RECORD==0:
            #Tensorboard
            if LEARNING_HAS_STARTED:
                #Record Reward
                mean_reward=float(acc_reward)/EPISODES_PER_RECORD
                sumOut=sess.run(re_sum,feed_dict={reward_summary:mean_reward})
                writer.add_summary(sumOut,episode)
                #Record loss:
                mean_loss=float(loss)/EPISODES_PER_RECORD
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
					state, reward=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
				env.render(close=True)
saver.save(sess,SAVE_PATH)
print "Model saved in path: ",SAVE_PATH
