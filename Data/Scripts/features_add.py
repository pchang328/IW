import sys
	
def perc_increase (before, after):
	return (after - before) / before 

def main(args):
	features_file = open('../Original/features_new.txt', 'rU')
	mass_before_index = 4
	mass_after_index = 5
	num_features_before_addition = 6 

	for line in features_file:
		arr = line.split()
		if len(arr) == num_features_before_addition:
			mass_before = float(arr[mass_before_index])
			mass_after = float(arr[mass_after_index])
			perc_incr = perc_increase(mass_before, mass_after)

			print line.rstrip() + "  " + str(perc_incr)

if __name__ == "__main__":
    main(sys.argv[1:])