#!/usr/bin/env python

import sys
import rospy
import random
import gazebo_parameters
from robot_ddpg_gazebo.srv import *

class gazebo_env:
	def __init__(self):
		# consts 
		self.MAX_VALUE=gazebo_parameters.MAX_VALUE
		self.MAX_X=gazebo_parameters.MAX_X
		self.NUM_OBSTACLES=gazebo_parameters.NUM_OBSTACLES
		self.MAX_TIME=gazebo_parameters.MAX_TIME
		self.INTERVAL_TIME=gazebo_parameters.INTERVAL_TIME
		self.NUM_VIAPOINTS=gazebo_parameters.NUM_VIAPOINTS
		self.OBSTACLE_NAMES=gazebo_parameters.OBSTACLE_NAMES
		self.PATH_REGULARIZATION_FACTOR=gazebo_parameters.PATH_REGULARIZATION_FACTOR
		self.obstacle_positions=self.reset()
	def step(self,action):
		#step, return reward 
		rospy.wait_for_service('env_loop_service')
		try:			
			client=rospy.ServiceProxy('env_loop_service',EnvLoopSrv)
			#float64[] viapoints;int16 num_viapoints;float32 max_time;float32 max_x;float32 interval_time;string[] obstacles;int16 num_obstacles;float32[] obstacle_positions
			response=client(action[0],self.NUM_VIAPOINTS,self.MAX_TIME,self.MAX_X,self.INTERVAL_TIME,self.OBSTACLE_NAMES,self.NUM_OBSTACLES,self.obstacle_positions)
			# State for next episode
			self.obstacle_positions=self.calculate_obstacle_positions()
			reward=0
			reward-=abs(action[0][0]) #Add first viapoint
			reward-=abs(action[0][self.NUM_VIAPOINTS-3]) #Add last viapoint
			for i in range(self.NUM_VIAPOINTS-3): #For each viapoint 
				reward-=abs(action[0][i+1]-action[0][i])
			reward*=PATH_REGULARIZATION_FACTOR
			reward-=response.reward
			return self.obstacle_positions,reward
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
	def reset(self):
		self.obstacle_positions=self.calculate_obstacle_positions()
		return self.obstacle_positions
	def calculate_obstacle_positions(self):
		obs_pos=[]
		for e in range(self.NUM_OBSTACLES):
			obs_pos.append(random.random()*self.MAX_X) #X
			obs_pos.append(random.random()*self.MAX_VALUE*2-self.MAX_VALUE) #Y
		return obs_pos
		

