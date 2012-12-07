import csv
import distribution
import random
import numpy as np
class1_data = distribution.symetricalNormalDistribution(100,0,2.25)
class2_data = distribution.asymetricalNormalDistribution(100,7,2.25,0.5)
fileobject = open('file_1.csv','wb')
csvfile = csv.writer(fileobject)
for val in range(len(class1_data)) :	
	csvfile.writerow([1,float(class1_data[val])])
for val in range(len(class2_data)) :
	csvfile.writerow([2,float(class2_data[val])])
fileobject.close()
