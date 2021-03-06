import yumi_env
#Environment
MAX_VALUE=5.0
MAX_X=10.0
NUM_OBSTACLES=3
MAX_TIME=5
INTERVAL_TIME=0.01
NUM_VIAPOINTS=5
OBSTACLE_NAMES=["obs_1","obs_2","obs_3"]
PATH_REGULARIZATION_FACTOR=0.5
OBSTACLE_RADIUS=0.75 #This value must be manually checked with the world you're using
X_BIAS=1 #Safety margin from TCP starting position for obstacles to appear
env=yumi_env.yumi_env()
ACTION_RANGE=env.MAX_VALUE
STATE_SIZE=env.NUM_OBSTACLES*2
ACTION_SIZE=env.NUM_VIAPOINTS-2
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
HIDDEN_SIZE_ACTOR=400
ACTOR_NAME="actor"
ACTOR_SUBSPACE_NAME="ACTOR_OPS"
ACTOR_TARGET_NAME="actor_target"
ACTOR_TARGET_SUBSPACE_NAME="TARGET_ACTOR_OPS"
#Critic hyperparameters
CRITIC_L2_WEIGHT_DECAY=1e-2
CRITIC_LEARNING_RATE=1e-3
HIDDEN_SIZE_CRITIC=400
CRITIC_NAME="critic"
CRITIC_SUBSPACE_NAME="CRITIC_OPS"
CRITIC_TARGET_NAME="critic_target"
CRITIC_TARGET_SUBSPACE_NAME="TARGET_CRITIC_OPS"
#Q function parameters:
DISCOUNT=0.99
#Algorithm parameters:
LEARNING_HAS_STARTED=False #Don't change this, it's a permanent flag
NUM_EPISODES=100000
EPOCHS_PER_EPISODE=1
WARMUP=0
EPISODES_PER_RECORD=100
TRAINING_ITERATIONS_PER_EPISODE=1
TAU=1e-2
NOISE_FACTOR=0.5
NOISE_MOD=0.999
OU_SPEED=0.15
OU_MEAN=0
OU_VOLATILITY=0.2
#Program parameters:
LOGS_PATH="/tmp/ddpg_yumi_logs"
SAVE_PATH="/tmp/ddpg_yumi_model.ckpt"
RESTORE_PREVIOUS_SESSION=True
VISUALIZE=True
EPISODE_CHECKPOINT=100
