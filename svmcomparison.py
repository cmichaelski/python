import urllib
import numpy as np
from numpy.linalg import inv
from sklearn import svm
from sklearn.metrics import accuracy_score
from random import randint
import operator
import copy
import math



def Get_data(file_handle, y_value, x_value):

	for line in file_handle:
		value = line.split()
		y_value.append(value[0])
		x_value.append([value[1],value[2]])

	return

def One_vs_One_data(y_value, x_value, new_y_value, new_x_value, numb_one, numb_two):

	for i in range(len(y_value)):
		if int(float(y_value[i]))==numb_one:
			new_y_value.append(1)
			new_x_value.append(x_value[i])
		if int(float(y_value[i]))==numb_two:
			new_y_value.append(-1)
			new_x_value.append(x_value[i])

	return		

def One_vs_All_data(y_value, x_value, new_y_value, new_x_value, numb):

	for i in range(len(y_value)):
		if int(float(y_value[i]))==numb:
			new_y_value.append(1)
			new_x_value.append(x_value[i])
		else:
			new_y_value.append(-1)
			new_x_value.append(x_value[i])

	return

def misclassified( W_non_numpy, Y_non_numpy, Solution):

	W = np.matrix(W_non_numpy)
	Y = np.matrix(Y_non_numpy)

	test_solution = np.sign(W * Solution.transpose() )

	Y_T = Y.transpose()

	mistakes = abs( test_solution - Y_T )
	numb_of_mistakes = (mistakes.sum())/2

	return numb_of_mistakes

def distance_between_points(x_one,x_two,y_one,y_two):

	square_distance = (x_one - y_one) ** 2  + (x_two - y_two) **2 
	distance = square_distance ** (0.5)

	return distance

def generate_random_points(number_of_points):

	sample = np.random.uniform(-1,1,(number_of_points,2))

	return sample

def generate_y_values(number_of_points, sample, y_value):

	for i in range(0,number_of_points):	
		y_value.append( np.sign(sample[i][1] - sample[i][0] + (0.25)*math.sin(math.pi*(sample[i][0]) ) ) )

	return

def Error_vector_calculation(Output_vector, Result_vector):

	Error_vector = abs( Output_vector - Result_vector.transpose() )
	error = ( Error_vector.sum())/2

	return error

def generate_llyod_centers(movement_in_centers, lloyd_centers, initialization_size, initialization_points, l_center_vector, new_dict_llyod ):

	throw_out = 0

	while movement_in_centers > 0.01 :
		dict_of_llyod_points={}
		for index in range(0,lloyd_centers):
			dict_of_llyod_points[index]=[0,[],[]]
		for j in range(0,initialization_size):
			distance_vector=[]
			for i in range(0,lloyd_centers):
				dist = distance_between_points(initialization_points[j][0],initialization_points[j][1], l_center_vector[i][0], l_center_vector[i][1] )
				distance_vector.append( dist )

				min_val, min_index = min( (val,idx) for (idx,val) in enumerate(distance_vector))

			if min_index in dict_of_llyod_points:
				dict_of_llyod_points[min_index][0] += 1
				dict_of_llyod_points[min_index][1].append( initialization_points[j][0] )
				dict_of_llyod_points[min_index][2].append( initialization_points[j][1] )

		new_center_vector={}

		for item in dict_of_llyod_points:
			if int(dict_of_llyod_points[item][0]) > 0:
				new_center_vector[item] = [ np.mean( dict_of_llyod_points[item][1] ) , np.mean( dict_of_llyod_points[item][2] ) ]
			else:
				new_center_vector[item] = [ l_center_vector[item][0], l_center_vector[item][1] ]

		delta_in_centers=[]

		for item in dict_of_llyod_points:
			delta_in_centers.append( distance_between_points(new_center_vector[item][0],new_center_vector[item][1], l_center_vector[item][0], l_center_vector[item][1] ) )

		movement_in_centers = sum( delta_in_centers )

		l_center_vector = copy.copy(new_center_vector)

	for item in dict_of_llyod_points:
		if int(dict_of_llyod_points[item][0]) == 0 :
			throw_out = 1
			print "we have issue"
			print item, dict_of_llyod_points[item][0]

	return throw_out


kernel_beats_regular = 0
number_of_runs = 0
counter = 0
initialization_size=100
test_set_size = 1000


while counter < 1000:
	y_value=[]
	y_out=[]
	dict_of_results={}

	throw_out_set = 0

	initialization_points = generate_random_points(initialization_size)
	test_out = generate_random_points(test_set_size)

	generate_y_values(initialization_size, initialization_points, y_value)
	generate_y_values(test_set_size, test_out, y_out)

	gamma=1.5



	clf = svm.SVC(gamma=1.5, kernel='rbf')
	clf.fit(initialization_points, y_value)	
			
	results = clf.predict(initialization_points)

	er_out = clf.predict(test_out)

	error_in_kernel = (1-accuracy_score(y_value,results)) * initialization_size

	error_out_kernel = (1-accuracy_score(y_out,er_out)) * test_set_size

	if error_in_kernel == 0:  #need to skip this run
		throw_out_set = 1
		print "throwing set out"


	new_dict_llyod={}

	movement_in_centers = 100

	lloyd_centers = 9
	l_center_vector = generate_random_points(lloyd_centers)

	throw_out = generate_llyod_centers(movement_in_centers, lloyd_centers, initialization_size, initialization_points, l_center_vector, new_dict_llyod )

	gamma = 1.5

	A=np.ones(shape=(initialization_size,lloyd_centers+1))

	for j in range(0,initialization_size):
		for i in range(0,lloyd_centers):
			A[j,i+1] = math.exp( -gamma * (distance_between_points(initialization_points[j][0],initialization_points[j][1], l_center_vector[i][0], l_center_vector[i][1] ) )**2 )

	Y = np.matrix(y_value)
	A_T = np.matrix( A.transpose() )
	First_part =  A_T * A 
	Second_part = Y * A 
	Inv_First_part = inv( First_part )
	Solution = Second_part * Inv_First_part

	Solution_vector = np.sign( A * Solution.transpose() )

	OUT=np.ones(shape=(test_set_size,lloyd_centers+1))

	for j in range(0,test_set_size):
		for i in range(0,lloyd_centers):
			OUT[j,i+1] = math.exp( -gamma * (distance_between_points(test_out[j][0],test_out[j][1], l_center_vector[i][0], l_center_vector[i][1] ) )**2  )

	OUT_vector = np.sign( OUT * Solution.transpose() )
	Y_OUT = np.matrix(y_out)

	error_in_regular = Error_vector_calculation(Solution_vector, Y)

	error_out_regular = Error_vector_calculation(OUT_vector, Y_OUT)

	if (throw_out+throw_out_set) == 0 :
		number_of_runs +=1
		if (error_out_regular >= error_out_kernel) :
			kernel_beats_regular +=1

	counter +=1

print 'Kernel beats Regular', kernel_beats_regular
print 'number_of_runs', number_of_runs

print 'Kernel beats Reg %', (float(kernel_beats_regular)/number_of_runs)*100


