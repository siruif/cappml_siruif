# Machine Learning PS1 | Problem A | Part 2
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu


import requests
import csv
import json


'''
This file genderizes people according to their names.
'''

def get_gender(name):

	url = "https://api.genderize.io/?name=" + name
	req = requests.get(url)
	result = json.loads(req.text)
	if len(result) > 0:
		return result['gender']

def genderize(input_data, output_filename):

	f = open(input_data, 'r')
	w = open(output_filename, 'w')

	writer = csv.writer(w)
	reader = csv.reader(f)

	for line in reader:
		if line[4] == '':
			name = line[1]
			gender = get_gender(name)
			if gender == "female":
				line[4] = "Female"
			else:
				line[4] = "Male"
		
		writer.writerow(line)

	f.close()
	w.close()