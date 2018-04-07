#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gazebo_parameters
import numpy as np
import csv
import time

# Loaded parameters:
env=gazebo_parameters.env


# Evaluator parameters:
EPISODES_TO_EVALUATE=1000

state_to_evaluate=[2,2,5,2,10,2]
action_to_evaluate=[[2.0,-2.0,4.0]]

f = open("out_0_0005.csv","wb")
cr = csv.writer(f,delimiter=';')
start = time.time()
for episode in range(EPISODES_TO_EVALUATE):
	env.state=state_to_evaluate
	reward=env.step(action_to_evaluate)
	cr.writerow([reward])
	if episode%100==0:
		print "episode ",episode
f.close()	
end = time.time()
print end-start


