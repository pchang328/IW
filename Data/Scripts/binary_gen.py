import sys

def avg(float_list):
	sum = 0 
	for num in float_list:
		sum += num
	return sum / (len(float_list))		

def main(args):
	read_file = open(args[0], 'rU')
	write_file = open(args[1], 'w')

	nums = []
	for line in read_file:
		arr = line.split()
		nums.append(float(arr[1]))		

	lo_avg = avg(nums)

	print lo_avg

	for lo in nums:
		if lo <= lo_avg:
			write_file.write('0 \n')
		else:
			write_file.write('1 \n')									

if __name__ == "__main__":
    main(sys.argv[1:])