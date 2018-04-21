#!/usr/bin/env python

import sys
import rospy
import random
import gazebo_parameters
import numpy as np
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
		self.OBSTACLE_RADIUS=gazebo_parameters.OBSTACLE_RADIUS
		self.X_BIAS=gazebo_parameters.X_BIAS
		self.Y_BIAS=gazebo_parameters.Y_BIAS
		self.obstacle_positions=self.reset()
		self.distance_covered=0
		self.obs_dis=0
	def step(self,action):
		action=np.array([np.concatenate([[0],action[0],[0]])])
		rospy.wait_for_service('env_loop_service')
		try:			
			client=rospy.ServiceProxy('env_loop_service',EnvLoopSrv)
			#float64[] viapoints;int16 num_viapoints;float32 max_time;float32 max_x;float32 interval_time;string[] obstacles;int16 num_obstacles;float32[] obstacle_positions
			response=client(action[0],self.NUM_VIAPOINTS,self.MAX_TIME,self.MAX_X,self.INTERVAL_TIME,self.OBSTACLE_NAMES,self.NUM_OBSTACLES,self.obstacle_positions)
			self.distance_covered=response.distance_covered
			self.obs_dis=response.obstacle_displacement
			self.reward=response.reward
			return response.reward
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
	def reset(self):
		self.obstacle_positions=self.calculate_obstacle_positions()
		return self.obstacle_positions
	def distance(self,x1,y1,x2,y2):
		vector=[x1-x2,y1-y2]
		return np.linalg.norm(vector)
	def calculate_obstacle_positions(self):
		obs_pos=[]
		obstacle_ok=False
		for e in range(self.NUM_OBSTACLES):
			# Make sure obstacles don't spawn inside each other
			while not obstacle_ok:
				# Calculate a sample position
				x_pos=random.random()*self.MAX_X
				if x_pos<self.X_BIAS:
					x_pos+=self.X_BIAS
				elif x_pos>self.MAX_X-self.X_BIAS:
					x_pos-=self.X_BIAS
				y_pos=random.random()*self.MAX_VALUE*2*self.Y_BIAS-self.MAX_VALUE*self.Y_BIAS
				obstacle_ok=True
				for i in range(len(obs_pos)/2):
					obstacle_ok*=self.distance(obs_pos[i*2],obs_pos[i*2+1],x_pos,y_pos)>self.OBSTACLE_RADIUS
			# New positions are valid
			obstacle_ok=False
			obs_pos.append(x_pos) #X
			obs_pos.append(y_pos) #Y
		return obs_pos
	def getDetailedResponse(self):
		return self.distance_covered, self.obs_dis, self.reward
	def setState(self,state_to_set):
		self.obstacle_positions=state_to_set
