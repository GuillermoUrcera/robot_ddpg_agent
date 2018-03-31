#!/usr/bin/env python

import sys
import rospy
import random
from robot_ddpg_gazebo.srv import *

class gazebo_env:
	def __init__(self):
		# consts TODO make these be loadable some other way
		self.MAX_VALUE=5
		self.MAX_X=10
		self.ITERATIONS_PER_CONFIG=100
		self.CONFIGS_PER_RUN=100
		self.NUM_OBSTACLES=3
		self.MAX_TIME=5
		self.INTERVAL_TIME=0.01
		self.NUM_VIAPOINTS=5
		self.OBSTACLE_NAMES=["obs_1","obs_2","obs_3"]
		self.obstacle_positions=self.reset()
	def step(self,action):
		#step, return reward 
		rospy.wait_for_service('env_loop_service')
		try:			
			client=rospy.ServiceProxy('env_loop_service',EnvLoopSrv)
			#float64[] viapoints;int16 num_viapoints;float32 max_time;float32 max_x;float32 interval_time;string[] obstacles;int16 num_obstacles;float32[] obstacle_positions
			response=client(action,self.NUM_VIAPOINTS,self.MAX_TIME,self.MAX_X,self.INTERVAL_TIME,self.OBSTACLE_NAMES,self.NUM_OBSTACLES,self.obstacle_positions)
			# State for next episode
			self.obstacle_positions=self.calculate_obstacle_positions()
			return self.obstacle_positions,response.reward
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
	def reset(self):
		self.obstacle_positions=self.calculate_obstacle_positions()
		return self.obstacle_positions
	def calculate_obstacle_positions(self):
		obs_pos=[]
		for e in range(self.NUM_OBSTACLES):
			obs_pos[e*2]=random.random()*self.MAX_X #X
			obs_pos[e*2+1]=random.random()*self.MAX_VALUE #Y
		return obs_pos
		

