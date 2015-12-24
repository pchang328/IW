import sys
from itertools import izip

def main(args):
	features_file = open('../features.txt', 'rU')
	lof_r_sc = open('../Original/lo_sc.txt', 'rU')
	hif_r_sc = open('../Original/hi_sc.txt', 'rU')
	lof_r_noise = open('../Original/lo_noise.txt', 'rU')
	hif_r_noise = open('../Original/hi_noise.txt', 'rU')

	combined_file = open('../data_new_float.txt', 'w')

	for a, b, c, d, e in izip(features_file, lof_r_sc, hif_r_sc, lof_r_noise, hif_r_noise):
		lo_sc = b.split()[1]
		hi_sc = c.split()[1]
		lo_noise = d.split()[1]
		hi_noise = e.split()[1]

		combined_file.write(a.rstrip() + "  " + str(lo_sc) + "  " + str(hi_sc) + "  " + str(lo_noise) + "  " + str(hi_noise) + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])