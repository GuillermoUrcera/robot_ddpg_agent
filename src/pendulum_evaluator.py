import pendulum_parameters
import tensorflow as tf
import numpy as np
import actor

# Loaded parameters:
env=pendulum_parameters.env
STATE_SIZE=pendulum_parameters.STATE_SIZE
ACTION_SIZE=pendulum_parameters.ACTION_SIZE
ACTOR_LEARNING_RATE=pendulum_parameters.ACTOR_LEARNING_RATE
ACTION_RANGE=pendulum_parameters.ACTION_RANGE
HIDDEN_SIZE_ACTOR=pendulum_parameters.HIDDEN_SIZE_ACTOR
ACTOR_NAME=pendulum_parameters.ACTOR_NAME
ACTOR_SUBSPACE_NAME=pendulum_parameters.ACTOR_SUBSPACE_NAME
MINIBATCH_SIZE=pendulum_parameters.MINIBATCH_SIZE
SAVE_PATH=pendulum_parameters.SAVE_PATH

# Evaluator parameters:
EPISODES_TO_EVALUATE=5
EPOCHS_PER_EPISODE_TO_EVALUATE=500
VISUALIZE_EVALUATOR=True

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
		if VISUALIZE_EVALUATOR:
			env.render()
		state, reward, done, info=env.step(my_actor.predict(np.reshape(state,(1,STATE_SIZE))))
	env.render(close=True)
