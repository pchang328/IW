import sys
from itertools import izip

def main(args):
	features_file = open('../features_with_horizontal.txt', 'rU')
	lof_r_sc = open('../Original/lo_sc_bin.txt', 'rU')
	hif_r_sc = open('../Original/hi_sc_bin.txt', 'rU')
	lof_r_noise = open('../Original/lo_noise_bin.txt', 'rU')
	hif_r_noise = open('../Original/hi_noise_bin.txt', 'rU')

	combined_file = open('../data_new_float_with_horizontal_bin.txt', 'w')

	for a, b, c, d, e in izip(features_file, lof_r_sc, hif_r_sc, lof_r_noise, hif_r_noise):
		lo_sc = b.split()[0]
		hi_sc = c.split()[0]
		lo_noise = d.split()[0]
		hi_noise = e.split()[0]

		combined_file.write(a.rstrip() + "  " + str(lo_sc) + "  " + str(hi_sc) + "  " + str(lo_noise) + "  " + str(hi_noise) + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])