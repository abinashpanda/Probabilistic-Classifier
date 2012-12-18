#!/usr/bin/python2.7 -tt
import math
import random
import numpy as np
from scipy import integrate
from scipy import optimize
import pylab
import csv
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
	prob_v_c1 = symetricGaussianDensity(data,mean_class1,sigma_class1)
	prob_v_c2 = asymetricGaussianDensity(data,mean_class2,sigma_class2,r_class2)
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
pi = 6.2832
def asymetricGaussianDensity(x,mean=0,sigma = 1,r = 1):
	sigma_squared = math.pow(sigma,2)
	scaling_factor = (2.0 / math.sqrt(pi)) * (1.0 / (sigma * (r+1)))
	x_squared = math.pow((x-mean),2)
	if x > mean :	
		x_squared = -x_squared/(sigma_squared)
		y = scaling_factor * math.exp(0.5 * x_squared)
	else : 
		x_squared = -x_squared/(sigma_squared*math.pow(r,2))	
		y = scaling_factor * math.exp(0.5 * x_squared)
	return y	
def symetricGaussianDensity(x,mean=0,sigma = 1):
	scaling_factor = 1.0/(math.sqrt(pi)*sigma)
	x_squared = math.pow((x-mean)/sigma,2)
	y = scaling_factor*math.exp(-0.5*x_squared)	
	return y
def symetricalNormalDistribution(size,mean = 0, sigma = 1):
	y = []
	x = np.arange(round((mean - (7*sigma)),3),round((mean + (7*sigma)),3),0.001)
	y_cum = []
	for val in x:
		y_cum.append(integrate.quad(symetricGaussianDensity,round((mean - (7*sigma)),3),float(val),args = (mean,sigma)))
	y_cum = np.array(y_cum)	
	diction_data = {}
	for val in range(len(x)):
		diction_data.update({round(float(y_cum[val,0]),3):round(float(x[val]),3)})
	uniformDistribution = []
	for val in range(size):
		uniformDistribution.append(round(random.random(),2))
	for val in range(len(uniformDistribution)):
		valid = diction_data.has_key(uniformDistribution[val])
		if valid :
			y.append(diction_data[uniformDistribution[val]])
	y = np.array(y)	
	return y
def asymetricalNormalDistribution(size,mean = 0,sigma = 1, r = 1):
	y = []
	x = np.arange(round((mean - (7*sigma)),3),round((mean + (7*sigma)),3),0.001)
	y_cum = []
	for val in x:
		y_cum.append(integrate.quad(asymetricGaussianDensity,round((mean - (7*sigma)),3),float(val),args = (mean,sigma,r)))
	y_cum = np.array(y_cum)	
	diction_data = {}
	for val in range(len(x)):
		diction_data.update({round(float(y_cum[val,0]),3):round(float(x[val]),3)})
	uniformDistribution = []
	for val in range(size):
		uniformDistribution.append(round(random.random(),3))
	for val in range(len(uniformDistribution)):
		valid = diction_data.has_key(uniformDistribution[val])
		if valid :
			y.append(diction_data[uniformDistribution[val]])	
	y = np.array(y)	
	return y
def probAsymGaussian(x,mean,sigma,r):
	return integrate.quad(asymetricGaussianDensity,round((mean - (7*sigma)),3),x,args = (mean,sigma,r))[0]
def probSymGaussian(x,mean,sigma):
	return integrate.quad(symetricGaussianDensity,round((mean - (7*sigma)),3),x,args = (mean,sigma))[0]
def gauss_likelyhood(mean,sigma,y):
	arr = np.power((y-mean),2)
	arr_sum = arr.sum()
	arr_sum = arr_sum/(2*sigma*sigma)
	arr_sum = arr_sum
	r = len(y)
	ret = r*((math.log(pi)/2) + (math.log(sigma)))
	ret = ret + arr_sum
	ret = ret
	return ret
def estimateGaussianParams(inputdata):
	sigma,mean = 1,0 
	mean_estimated = optimize.fmin(lambda m: gauss_likelyhood(m,sigma,inputdata),mean)
	sigma_estimated = optimize.fmin(lambda s: gauss_likelyhood(mean_estimated,s,inputdata),sigma)
	mean_estimated = round(mean_estimated,2)
	sigma_estimated = round(sigma_estimated,2)
	return mean_estimated,sigma_estimated	
def sgn_arr(y,thresh):
	y_sgn = []
	for val in y :
		if (float(val) >= float(thresh)):
			y_sgn.append(1)
		else :
			y_sgn.append(0)
	y_sgn = np.array(y_sgn)
	y_invert = (1-y_sgn)
	return y_sgn,y_invert
def asymgauss_likelyhood(mean,sigma,r,inputdata):
	arr = np.power((inputdata-mean),2)
	y_sgn,y_invert = sgn_arr(inputdata,mean)
	arr2 = ((y_sgn*arr)/(2*sigma*sigma)) + ((y_invert*arr)/(2*sigma*sigma*r*r))
	arr_sum = arr2.sum()
	length = len(inputdata)
	ret = ((math.log(pi)/2) + (math.log(sigma))+ math.log(r+1) - math.log(2))
	ret = ret*length	
	ret = ret + arr_sum
	return ret
def estimateAsymGaussianParams(inputdata):
	mean_estimated = 0
	sigma_estimated = 1
	r_estimated = 1
	for val in range(50): 
		mean_estimated = optimize.fmin(lambda m: asymgauss_likelyhood(m,sigma_estimated,r_estimated,inputdata),mean_estimated)
		sigma_estimated = optimize.fmin(lambda s : asymgauss_likelyhood(mean_estimated,s,r_estimated,inputdata),sigma_estimated)
		r_estimated = optimize.fmin(lambda r : asymgauss_likelyhood(mean_estimated,sigma_estimated,r,inputdata),r_estimated)
	mean = round(mean_estimated,2)
	sigma = round(sigma_estimated,2)
	r = round(r_estimated,2)
	return mean,sigma,r

