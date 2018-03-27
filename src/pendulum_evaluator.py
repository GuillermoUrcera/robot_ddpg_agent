import gym
import actor
import critic
import tensorflow as tf
import numpy as np


#Actor hyperparameters
ACTOR_LEARNING_RATE=1e-4
HIDDEN_SIZE_ACTOR=8
ACTOR_NAME="actor"
ACTOR_SUBSPACE_NAME="ACTOR_OPS"
ACTOR_TARGET_NAME="actor_target"
ACTOR_TARGET_SUBSPACE_NAME="TARGET_ACTOR_OPS"
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

tf.reset_default_graph()
sess=tf.Session()


with tf.variable_scope(ACTOR_SUBSPACE_NAME):
    my_actor=actor.Actor(sess,STATE_SIZE,ACTION_SIZE,ACTOR_LEARNING_RATE,ACTION_RANGE,HIDDEN_SIZE_ACTOR,ACTOR_NAME,ACTOR_SUBSPACE_NAME,MINIBATCH_SIZE)

saver = tf.train.Saver()


# Restore variables from disk.
saver.restore(sess, "/tmp/model.ckpt")
for e in range(5):
	state=env.reset()
	for i in range(500):
		env.render()
		state, reward, done, info=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
	env.render(close=True)
