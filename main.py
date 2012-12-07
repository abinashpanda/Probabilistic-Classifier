import distribution
import csv
import numpy as np
import math
def getInputData(filename):
	fileobject = open(filename,'rb')
	csvfile = csv.reader(fileobject)
	class1_data = []
	class2_data = []
	for row in csvfile:
		class_data = int(row[0])
		if class_data == 1:
			class1_data.append(float(row[1]))
		else :
			class2_data.append(float(row[1]))
	class1_data = np.array(class1_data)
	class2_data = np.array(class2_data)
	fileobject.close()	
	return class1_data,class2_data
def computeProb(data,n1,mean_class1,sigma_class1,n2,mean_class2,sigma_class2,r_class2):
	prob_v_c1 = distribution.probSymGaussian(data,mean_class1,sigma_class1)
	prob_v_c2 = distribution.probAsymGaussian(data,mean_class2,sigma_class2,r_class2)
	prob1 = (n1*prob_v_c1)/((n1*prob_v_c1) + (n2*prob_v_c2))
	prob2 = (n2*prob_v_c2)/((n1*prob_v_c1) + (n2*prob_v_c2))
	prob1 = round(prob1,2)
	prob2 = round(prob2,2)	
	return prob1,prob2
def classPrediction(prob1,prob2):
	if prob1 > (prob2): 
		return 1
	else :
		return 2	
	
if __name__ == '__main__':
	class1_data,class2_data = getInputData('file_1.csv')
	mean_class1,sigma_class1 = distribution.estimateGaussianParams(class1_data)
	mean_class2,sigma_class2,r_class2 = distribution.estimateAsymGaussianParams(class2_data)
	n1 = len(class1_data)
	n2 = len(class2_data)
	count = 0
	for val in class1_data:
		prob1,prob2 = computeProb(float(val),n1,mean_class1,sigma_class1,n2,mean_class1,sigma_class2,r_class2)		
		class_predicted = classPrediction(prob1,prob2)
		if class_predicted == 1:
			count = count + 1
	for val in class2_data:
		prob1,prob2 = computeProb(float(val),n1,mean_class1,sigma_class1,n2,mean_class1,sigma_class2,r_class2)		
		class_predicted = classPrediction(prob1,prob2)
		if class_predicted == 2:
			count = count + 1
	print "Mean and sigma of class1 (symetrical gaussian distribution) : ",mean_class1 , sigma_class1
	print "Mean, sigma, r of class2 (asymetrical gaussian distribution): ",mean_class2 , sigma_class2 , r_class2
	print "Number of elements in class1 ",n1
	print "Number of elements in class2 ",n2
	print 'Accuracy is :',(count/(float(n1+n2)))*100,'%'
