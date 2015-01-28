import urllib
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from random import randint
import operator
import copy
from numpy.linalg import inv
import math

import matplotlib.pyplot as plt


def generate_random_points(number_of_points):

	sample = np.random.uniform(-1,1,(number_of_points,2))

	return sample

def generate_line():

	left_side = np.random.uniform(-1,1,1)
	#(-1,left_side) first point
	right_side = np.random.uniform(-1,1,1)
	#(1,right_side) second point

	slope = (right_side - left_side) / 2
	b_intercept = (right_side + left_side) / 2

	line = [slope, b_intercept, left_side, right_side] 

	return line


x_zero = []
x_one= []
y_zero =[]
y_one =[]
line=[]
results_vector=[]
sample_length = 100

sample=generate_random_points(sample_length)

line=generate_line()

for i in range(0,sample_length):
	if ( sample[i][1] > ( float(line[0])* sample[i][0] + float(line[1]) ) ) :
		x_zero.append( sample[i][0] )
		x_one.append( sample[i][1] )
		results_vector.append( [ sample[i][0], sample[i][1], 1 ] )
	else:
		y_zero.append( sample[i][0] )
		y_one.append( sample[i][1] )
		results_vector.append( [ sample[i][0], sample[i][1], -1 ] )

#preceptron learning

iter_count =0
tolerance =1000
weight_vector = np.array([0,0,0])
eta = 0.1

while iter_count < tolerance:
	misclassified =[]
	permutation = np.random.permutation(sample_length)
	for number in permutation:
		point = np.array( [1, results_vector[number][0], results_vector[number][1] ])
		classififed_point = np.sign( np.inner( weight_vector, point) )
		if (abs(classififed_point- results_vector[number][2]) > 0.1 ):
			misclassified.append( classififed_point)
			weight_vector  = np.add( weight_vector,  eta* results_vector[number][2] * point)
	if len(misclassified) ==0 :
		iter_count = tolerance
	iter_count +=1

percep_leftside = (weight_vector[1] - weight_vector[0])/weight_vector[2]
percep_rightside = 	(-weight_vector[1] - weight_vector[0])/weight_vector[2]

plt.plot([-1,1],[line[2],line[3]],[-1,1],[percep_leftside,percep_rightside],x_zero, x_one, 'ro', y_zero, y_one, 'bo')
plt.show()

