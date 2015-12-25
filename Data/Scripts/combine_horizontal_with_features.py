import sys
from itertools import izip

def list_to_string (l):
    x = ""
    for elt in l:
        x = x + str(elt) + " "
    return x

def main(args):
	features_file = open('../features.txt', 'rU')
	features_horizontal = open('../sorted_horizontal.txt', 'rU')

	d_features_horizontal = {}

	for line in features_horizontal:
		arr = line.split()
		name = arr[0]
		d_features_horizontal[name] = list_to_string(arr[1:])

	for line in features_file:
		arr = line.split()
		name = arr[0]
		if name in d_features_horizontal.keys():
			print line.strip() + " " + d_features_horizontal[name]
		elif name[:-1] in d_features_horizontal.keys():
			print line.strip() + " " + d_features_horizontal[name[:-1]]
		else:
			print line.strip()


if __name__ == "__main__":
    main(sys.argv[1:])