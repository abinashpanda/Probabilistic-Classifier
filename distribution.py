#!/usr/bin/python2.7 -tt
import math
import random
import numpy as np
from scipy import integrate
from scipy import optimize
import pylab
def asymetricGaussianDensity(x,mean=0,sigma2 = 1,r = 1):
	x_squared = math.pow((x-mean),2)
	if x > mean :	
		x_squared = -x_squared/(2*sigma2)
	else : 
		x_squared = -x_squared/(2*sigma2*r*r)	
	y = math.exp(x_squared)
	y = (2*y)/(((math.sqrt(6.28*sigma2))*(r+1)))
	return y
def symetricGaussianDensity(x,mean=0,sigma2 = 1):
	x_squared = math.pow((x-mean),2)
	x_squared = -x_squared/(2*sigma2)	
	y = math.exp(x_squared)
	y = y/(math.sqrt(6.28*sigma2))
	return y
def symetricalNormalDistribution(size,mean = 0, sigma2 = 1):
	y = []
	x = np.arange(round((mean - (3*math.sqrt(sigma2))),3),round((mean + (3*math.sqrt(sigma2))),3),0.001)
	y_cum = []
	for val in x:
		y_cum.append(integrate.quad(symetricGaussianDensity,round((mean - (3*math.sqrt(sigma2))),3),float(val),args = (mean,sigma2)))
	y_cum = np.array(y_cum)	
	diction_data = {}
	for val in range(len(x)):
		diction_data.update({round(float(y_cum[val,0]),3):round(float(x[val]),3)})
	uniformDistribution = []
	for val in range(size):
		uniformDistribution.append(round(random.random(),2))
	for val in range(len(uniformDistribution)):
		comp = diction_data.has_key(uniformDistribution[val])
		if comp :
			y.append(diction_data[uniformDistribution[val]])
	y = np.array(y)	
	return y
def asymetricalNormalDistribution(size,mean = 0,sigma2 = 1, r = 1):
	y = []
	x = np.arange(round((mean - (3*math.sqrt(sigma2))),3),round((mean + (3*math.sqrt(sigma2))),3),0.001)
	y_cum = []
	for val in x:
		y_cum.append(integrate.quad(asymetricGaussianDensity,round((mean - (3*math.sqrt(sigma2))),3),float(val),args = (mean,sigma2,r)))
	y_cum = np.array(y_cum)	
	diction_data = {}
	for val in range(len(x)):
		diction_data.update({round(float(y_cum[val,0]),3):round(float(x[val]),3)})
	uniformDistribution = []
	for val in range(size):
		uniformDistribution.append(round(random.random(),2))
	for val in range(len(uniformDistribution)):
		comp = diction_data.has_key(uniformDistribution[val])
		if comp :
			y.append(diction_data[uniformDistribution[val]])	
	y = np.array(y)	
	return y	
def probAsymGaussian(x,mean,sigma2,r):
	return integrate.quad(asymetricGaussianDensity,round((mean - (3*math.sqrt(sigma2))),3),x,args = (mean,sigma2,r))[0]
def probSymGaussian(x,mean,sigma2):
	return integrate.quad(symetricGaussianDensity,round((mean - (3*math.sqrt(sigma2))),3),x,args = (mean,sigma2))[0]
def gauss_likelyhood(mean,sigma2,y):
	arr = np.power((y-mean),2)
	arr_sum = arr.sum()
	arr_sum = arr_sum/(2*sigma2)
	arr_sum = arr_sum
	r = len(y)
	ret = r*((math.log(6.28)/2) + (math.log(sigma2)/2))
	ret = ret + arr_sum
	ret = ret
	return ret
def estimateGaussianParams(inputdata):
	sigma,mean = 1,0 
	mean_estimated = optimize.fmin(lambda m: gauss_likelyhood(m,sigma,inputdata),mean)
	sigma_estimated = optimize.fmin(lambda s: gauss_likelyhood(mean_estimated,s,inputdata),sigma)
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
def asymgauss_likelyhood(mean,sigma2,r,inputdata):
	arr = np.power((inputdata-mean),2)
	y_sgn,y_invert = sgn_arr(inputdata,mean)
	arr2 = ((y_sgn*arr)/(2*sigma2)) + ((y_invert*arr)/(2*sigma2*r*r))
	arr_sum = arr2.sum()
	length = len(inputdata)
	ret = ((math.log(6.28)/2) + (math.log(sigma2)/2)+ math.log(r+1) - math.log(2))
	ret = ret*length	
	ret = ret + arr_sum
	ret = ret
	return ret
def estimateAsymGaussianParams(inputdata):
	mean_estimated = 0
	sigma2_estimated = 1
	r_estimated = 1
	for val in range(50): 
		mean_estimated = optimize.fmin(lambda m: asymgauss_likelyhood(m,sigma2_estimated,r_estimated,inputdata),mean_estimated)
		sigma2_estimated = optimize.fmin(lambda s : asymgauss_likelyhood(mean_estimated,s,r_estimated,inputdata),sigma2_estimated)
		r_estimated = optimize.fmin(lambda r : asymgauss_likelyhood(mean_estimated,sigma2_estimated,r,inputdata),r_estimated)
	mean = round(mean_estimated,2)
	sigma2 = round(sigma2_estimated,2)
	r = round(r_estimated,2)
	return mean,sigma2,r
if __name__ == '__main__':
	inputdata = asymetricalNormalDistribution(100,0,2,0.5)
	mean,sigma2,r = estimateAsymGaussianParams(inputdata)	
	print mean,sigma2,r	
	inputdata2 = symetricalNormalDistribution(15,-3,1.5)
	mean,sigma2 = estimateGaussianParams(inputdata2)
	print mean,sigma2
