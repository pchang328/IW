import skimage.io as io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from skimage import data, img_as_float
from skimage import exposure
from scipy.optimize import curve_fit
import math

def show_images(images,titles=None) :
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def gen_grey_image(image_file) :
    image = io.imread(image_file)
    gray_image = rgb2gray(image)
    # io.imshow(gray_image)
    # io.show()
    # print "Colored image shape:\n", image.shape
    # print "Grayscale image  shape:\n", gray_image.shape
    return gray_image
# show_images(images=[image,gray_image],
#             titles=["Color","Grayscale"])

def avg_float_list (float_list) :
    sum = 0
    for i in float_list:
        sum += i 
    return (sum / len(float_list))

def average_darkest_pixel (images) :
    sum = 0
    for img in images:
        sum += img[img > 0].min()
    return (sum / len(images))

def gen_avg_intensity_list_row (gray_img) :
    avg_intensities = []
    for row in range (gray_img.shape[0]):
        row_to_avg = []
        for col in range (gray_img.shape[1]):
            pixel_intensity = gray_img[row][col]
            if pixel_intensity != 0 : 
                if pixel_intensity > 1: row_to_avg.append(1)
                else: row_to_avg.append(pixel_intensity)
        if len(row_to_avg) != 0: avg_intensities.append(avg_float_list(row_to_avg))    
    return avg_intensities        

def gen_avg_intensity_list_col (gray_img) :
    avg_intensities = []
    for col in range (gray_img.shape[1]):
        col_to_avg = []
        for row in range (gray_img.shape[0]):
            pixel_intensity = gray_img[row][col]
            if pixel_intensity != 0 : 
                if pixel_intensity > 1: col_to_avg.append(1)
                else: col_to_avg.append(pixel_intensity)
        if len(col_to_avg) != 0: avg_intensities.append(avg_float_list(col_to_avg))    
    return avg_intensities            

def plot_intensity_list (avg_intensities) : 
    plt.plot(avg_intensities)
    plt.ylabel('intensity')

def normalize(img, const):
    # Determine scaling factor first 
    darkest_pixel = img[img > 0].min()
    scale = const / darkest_pixel
    return np.multiply(img, scale)

def crop (image):
    row_cut_off = 700
    return image[0:700, :]

def exp_func(x, a, b, c):
    return a * np.exp(-b * (x)) + c

def exp_func_up(x, a, b):
    return a * np.exp(b * x)

# Divides the reed into the edges and the center
def divide_horizontal(col_int_list):
    length = len(col_int_list)
    l_edge_idx_end = int(math.floor(0.25 * length))
    right_idx_start = int(math.floor(0.75 * length))
    return (col_int_list[:l_edge_idx_end], col_int_list[l_edge_idx_end:right_idx_start], 
        col_int_list[right_idx_start:])
    

def main(argv):
    list_of_files = glob.glob('../Reed_Pics/*.png')
    intensity_dist_files = []
    file_to_image = {}

    # Generate mapping between file name to gray scaled, cropped image matrix 
    for file in list_of_files:
        gray_img = gen_grey_image(file) 
        file_to_image[file] = crop(gray_img) 

    const = average_darkest_pixel(file_to_image.values())               

    # Normalize your images matrices to the specified const 
    for file, img in file_to_image.iteritems():    
        # io.imshow(img)
        # io.show()
        file_to_image[file] = normalize(img, const / 2)
        # io.imshow(normalize(img, const))
        # io.show()

    if argv[0] == "-r":
        for name, img in file_to_image.iteritems():
            avg_intensity_list = gen_avg_intensity_list_row(img)
            xdata = np.array(range(len(avg_intensity_list)))
            ydata = np.array(avg_intensity_list)
            popt, pcov = curve_fit(exp_func, xdata, ydata)
            perr = np.sqrt(np.diag(pcov))
            plt.plot(exp_func(xdata, *popt))
            plot_intensity_list(avg_intensity_list)

    if argv[0] == "-c":
        for img in file_to_image.values():
            avg_intensity_list = gen_avg_intensity_list_col(img)
            (l, m, r) = divide_horizontal(avg_intensity_list)

            # xdata = np.array(range(len(l)))
            # ydata = np.array(l)
            # popt, pcov = curve_fit(exp_func, xdata, ydata)
            # plt.plot(exp_func(xdata, *popt))
            # plot_intensity_list(l) 
            # plt.show()

            fx = np.array(range(len(r)))
            fy = np.array(r)

            norm_x = fx.min()
            norm_y = fy.max()
            fx2 = fx - norm_x + 1
            fy2 = fy/norm_y

            popt, pcov = curve_fit(exp_func_up, fx2, fy2)
            plt.plot(exp_func_up(fx, *popt))
            plot_intensity_list(fy) 
            plt.show()

            # plot_intensity_list(avg_intensity_list) 
            # plt.show()      
            # plot_intensity_list(l) 
            # plt.show()    
            # plot_intensity_list(m) 
            # plt.show()    
            # plot_intensity_list(r) 
            # plt.show()    
                 

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])









