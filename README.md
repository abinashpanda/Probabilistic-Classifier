README
======

###distribution.py

It is the module that consists of all the functions required for generation to parameter estimation of both symetric as well as asymetric gaussian distribution.
1. symetricalNormalDistribution : it generates random number according to symetric gaussian distribution (given the number of values to generated, mean and sigma of the distribution)
2. symetricalNormalDistribution : it generates random number according to asymetric gaussian distribution (given the number of values to generated, mean, sigma and r of the distribution)
3.estimateGaussianParams : it estimates the parameters of the gaussian distribution using maximum likelyhood estimate
4.estimateAsymGaussianParams : it estimates the parameters of the asymetric gaussian distribution using maximum likelyhood estimate


###fileGenerator.py
This program generates the .csv file consisting of the class of the patient and their linear score representing their medical data.

###main.py
This is the main program where 
1. the data from the .csv file are read
2. the parameter estimation is done
3. the probability of the patient having cancer is computed using Bayesian Classifier.

The data consists of two class of patients where <code>class1_data</code> represent the class of patients having cancer and <code>class2_data</code> represent the class of patients that don't have cancer.

