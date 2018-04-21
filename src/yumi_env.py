#!/usr/bin/env python

import rospy
import yumi_parameters
from robot_ddpg_yumi.srv import *

class yumi_env:
	def __init__(self):
		# consts 
		self.MAX_VALUE=yumi_parameters.MAX_VALUE
		self.MAX_X=yumi_parameters.MAX_X
		self.NUM_OBSTACLES=yumi_parameters.NUM_OBSTACLES
		self.MAX_TIME=yumi_parameters.MAX_TIME
		self.INTERVAL_TIME=yumi_parameters.INTERVAL_TIME
		self.NUM_VIAPOINTS=yumi_parameters.NUM_VIAPOINTS
		self.OBSTACLE_NAMES=yumi_parameters.OBSTACLE_NAMES
		self.PATH_REGULARIZATION_FACTOR=yumi_parameters.PATH_REGULARIZATION_FACTOR
		self.OBSTACLE_RADIUS=yumi_parameters.OBSTACLE_RADIUS
		self.X_BIAS=yumi_parameters.X_BIAS
                self.obstacle_positions=self.getState()
                rospy.init_node('yumi_env', anonymous=True)
	def step(self,action):
		action=np.array([np.concatenate([[0],action[0],[0]])])
		rospy.wait_for_service('yumi_env_loop_service')
		try:
			client=rospy.ServiceProxy('yumi_env_loop_service',EnvLoopSrv)
			response=client(action[0],self.NUM_VIAPOINTS,self.MAX_TIME,self.MAX_X,self.INTERVAL_TIME,self.OBSTACLE_NAMES,self.NUM_OBSTACLES,self.obstacle_positions)
			return response.reward
		except rospy.ServiceException, e:
			print "Loop service call failed: %s"%e
	def getState(self):
		obs_pos=[]
		try:
			client=rospy.ServiceProxy('yumi_env_state_service',EnvStateSrv)
			response=client()
			if response.num_obstacles!=self.NUM_OBSTACLES:
				print "Error: incorrect number of obstacles!"
			else:
				for obstacle in range(response.num_obstacles):
					obs_pos.append(response.obstacle_positions[obstacle*2])   #X
					obs_pos.append(response.obstacle_positions[obstacle*2+1]) #Y
		except rospy.ServiceException, e:
			print "State service call failed: %s"%e
		return obs_pos
			
