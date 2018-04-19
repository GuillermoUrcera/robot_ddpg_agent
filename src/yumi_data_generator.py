#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import yumi_parameters
import numpy as np
import csv
import random

# Loaded parameters:
env=yumi_parameters.env
ACTION_RANGE=yumi_parameters.ACTION_RANGE
ACTION_SIZE=yumi_parameters.ACTION_SIZE

# Evaluator parameters:
EPISODES_TO_EVALUATE=1000


f = open("yumi_generated_data.csv","wb")
cr = csv.writer(f,delimiter=';')
for episode in range(EPISODES_TO_EVALUATE):
	raw_input("Press Enter to continue...")# This allows you to reset the experiment
	#Generate random action
	action_to_evaluate=[]
        for e in range(ACTION_SIZE):
		action_to_evaluate.append(random.random())
	action_to_evaluate=np.array([action_to_evaluate])
	#Get state
	state=env.getState()
	#Get reward
	reward=env.step(action_to_evaluate)
	#Record interaction
	cr.writerow([state,action,reward])
	if episode%100==0:
		print "episode ",episode
f.close()	
