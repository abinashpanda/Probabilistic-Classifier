import csv
import distribution
class1_data = distribution.symetricalNormalDistribution(15,-3, 2.25)
class2_data = distribution.asymetricalNormalDistribution(25,1,1.5,0.5)
data = {}
for val in class1_data:
	data.update({float(val):1})
for val in class2_data:
	data.update({float(val):2})
fileobject = open('file_1.csv','wb')
csvfile = csv.writer(fileobject)
#csvfile.writerow(['class','value'])
for val in data.keys():
	csvfile.writerow([data[val],float(val)])
fileobject.close()

