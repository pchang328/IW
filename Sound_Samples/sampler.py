import sys
import re
from os import listdir
from os.path import isfile, join
import random
import playwav
import time

def create_file_to_sc_dict():
	hi_truth_sc_file = open('./Data/spec_centroids_hi_final_new.txt')
	lo_truth_sc_file = open('./Data/spec_centroids_lo_final_new.txt')
	d = {}
	for line in hi_truth_sc_file:
		arr = line.split()		
		d[arr[0]] = float(arr[1])
	for line in lo_truth_sc_file:
		arr = line.split()		
		d[arr[0]] = float(arr[1])
	return d

def create_file_to_noise_dict():
	hi_truth_noise_file = open('./Data/spec_centroids_hi_noise.txt')
	lo_truth_noise_file = open('./Data/spec_centroids_lo_noise.txt')
	d = {}
	for line in hi_truth_noise_file:
		arr = line.split()		
		d[arr[0]] = float(arr[1])
	for line in lo_truth_noise_file:
		arr = line.split()		
		d[arr[0]] = float(arr[1])
	return d

def choose_rand_pair(hi):
	rand1 = random.randint(0,hi)
	rand2 = random.randint(0,hi)
	while rand1 == rand2:
		rand2 = random.randint(0,hi)
	return (rand1, rand2)		

def main(args):
	sound_path_hi = "./Sounds/High/"
	sound_path_lo = "./Sounds/Low/"
	onlyfiles_high = [f for f in listdir(sound_path_hi) if isfile(join(sound_path_hi, f))]
	onlyfiles_lo = [f for f in listdir(sound_path_lo) if isfile(join(sound_path_lo, f))]

	r = re.compile('.wav$')
	high_files = filter (r.search, onlyfiles_high)
	lo_files = filter(r.search, onlyfiles_lo)

	# This contains mapping from pairs of recordings to 1 or 2. 1 means first recording 
	# was pereceived to be brighter, 2 means second recording was perceieved to be brigher
	d = {}; query_string = ""; truth_table = {}; error = 0;

	if args[1] == '-s':
		# The string for input
		query_string = "1 if first is brighter, 2 if second brighter, 3 if same: "		
		truth_table = create_file_to_sc_dict()
	else:
		query_string = "1 if first is noisier, 2 if second is noisier, 3 if same: "
		truth_table = create_file_to_noise_dict()
		error = 0.02

	for i in range(int(args[0])):
		(file_index_1, file_index_2) = choose_rand_pair(len(high_files) - 1)
		file1 = ""
		file2 = ""
		if (random.uniform(0,1) <= 1):
			file1 = lo_files[file_index_1]
			file2 = lo_files[file_index_2]
			wav1 = sound_path_lo + file1
			wav2 = sound_path_lo + file2
			playwav.play(wav1)
			time.sleep(2)
			playwav.play(wav2)
		else:
			file1 = high_files[file_index_1]
			file2 = high_files[file_index_2]
			wav1 = sound_path_hi + file1
			wav2 = sound_path_hi + file2
			playwav.play(wav1)
			time.sleep(2)
			playwav.play(wav2)

		user_input = raw_input(query_string)
		while user_input != "1" and user_input != "2" and user_input != "3":
			user_input = raw_input("Wrong input\n" + query_string)

		d[(file1,file2)] = int(user_input)

	# Determining accuracy of using spectral centroids
	correct = 0
	for pair, result in d.iteritems():
		t1 = truth_table[pair[0]]
		t2= truth_table[pair[1]]
		print pair
		print t1
		print t2
		print result
		if abs(t1 - t2) <= error and result == 3:
			print "error" + str(error)
			print abs(t1 - t2)
			correct = correct + 1
		elif t1 > t2 and result == 1: 
			correct = correct + 1
		elif t2 > t1 and result == 2:
			correct = correct + 1

	print "Percentage Correct: %f" % (correct / float(len(d)))


if __name__ == "__main__":
    main(sys.argv[1:])