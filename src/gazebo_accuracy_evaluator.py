#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gazebo_parameters
import numpy as np
import csv

# Loaded parameters:
env=gazebo_parameters.env


# Evaluator parameters:
EPISODES_TO_EVALUATE=10

state_to_evaluate=[2,2,5,2,10,2]
action_to_evaluate=[[2.0,-2.0,4.0]]

f = open("out.csv","wb")
cr = csv.writer(f,delimiter=';')
for e in range(EPISODES_TO_EVALUATE):
	env.state=state_to_evaluate
	reward=env.step(action_to_evaluate)
	cr.writerow([reward])
f.close()	


