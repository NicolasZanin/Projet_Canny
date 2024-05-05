#!/usr/bin/env python3
import argparse
import numpy as np
from numba import cuda
import numba as nb
from PIL import Image
import sys
import math

input_image = None
output_image = None
apply_args = [True, True, True, True, True]
thread_block_size = (8, 8)  # Default thread block size

# Calculate Blocks par grid necessary for the source image depending on thread_block_size
def calcultate_blocks_per_grid(source):
    width, height = source.shape[0], source.shape[1]
    blocks_per_grid_x = math.ceil(width / thread_block_size[0])
    blocks_per_grid_y = math.ceil(height / thread_block_size[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return blocks_per_grid

# Convert color to gray with GPU
@cuda.jit
def RGB_to_BW_kernel(source, destination, offset):
    width, height = source.shape[:2]
    x, y = cuda.grid(2)

    # Verify if the absolute position is smaller than the width and height of source image
    if (x < width and y < height):
        # Creates an offset to manage edges
        r_x = (x + offset) % width 
        r_y = (y + offset) % height

        # Formula to convert color in gray
        destination[r_x, r_y] = np.uint8(0.3 * source[r_x, r_y, 0] + 0.59 * source[r_x, r_y, 1] + 0.1 * source[r_x, r_y, 2])

# Perform the GPU Kernel BW
def gpu_rgb_to_bw(source, threads_per_block, blocks_per_grid):
    width, height = source.shape[:2]
    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array((width, height), dtype=np.uint8)

    for offset in range(33, 1, -1):

        # Here is 6 iterations by default, but if we raise this number, that upgrade the image quality in output
        for i in range(6):
            RGB_to_BW_kernel[blocks_per_grid, threads_per_block](device_input_image, device_output_image, offset)
            cuda.synchronize()

    output = device_output_image.copy_to_host()
    return output

# Perform Gaussian blurring kernel on a single pixel
@cuda.jit
def gaussian_blur_kernel(source, destination, filter):
    x, y = cuda.grid(2)
    size_filter_x, size_filter_y = filter.shape[:2]
    
    if x < source.shape[0] and y < source.shape[1]:
        width, height = source.shape

        # Apply the convolution filter on a single pixel
        weighted_sum = 0
        normalization_factor = 0
        
        for index_filter_x in range(size_filter_x):
            for index_filter_y in range(size_filter_y):
                source_x = x + index_filter_x - size_filter_x // 2
                source_y = y + index_filter_y - size_filter_y // 2
                
                if source_x >= 0 and source_x < width and source_y >= 0 and source_y < height:
                    pixel_value = source[source_x, source_y]
                else:
                    pixel_value = source[x, y]
                
                weighted_sum += pixel_value * filter[index_filter_x][index_filter_y]
                normalization_factor += filter[index_filter_x][index_filter_y]
        
        if normalization_factor != 0:
            destination[x, y] = int(math.ceil(weighted_sum // normalization_factor))
        else:
            destination[x, y] = 0

# Perform the GPU Kernel gaussian blur
def gpu_gaussian_blur(image, threads_per_block, blocks_per_grid):
    filter = np.array([[1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1]])

    device_input_image = cuda.to_device(image)
    device_output_image = cuda.device_array_like(image)
    device_filter = cuda.to_device(filter)

    gaussian_blur_kernel[blocks_per_grid, threads_per_block](device_input_image, device_output_image, device_filter)
    return device_output_image.copy_to_host()

# Perform sobel kernel on a single pixel
@cuda.jit
def sobel_kernel(source, destination, sobel_x, sobel_y, clamped_magnitude):
    x, y = cuda.grid(2)
    size_sobel_x = sobel_x.shape

    if x < source.shape[0] and y < source.shape[1]:
        # Apply the sobel mask on a single pixel
        width, height = source.shape[:2]
        gradient_x = 0
        gradient_y = 0

        for index_sobel_x in range(size_sobel_x[0]):
            for index_sobel_y in range(size_sobel_x[1]):
                source_x = x + index_sobel_x - size_sobel_x[0] // 2
                source_y = y + index_sobel_y - size_sobel_x[1] // 2
                
                if source_x >= 0 and source_x < width and source_y >= 0 and source_y < height:
                    pixel_value = source[source_x, source_y]
                else:
                    pixel_value = source[x, y]  

                gradient_x += pixel_value * sobel_x[index_sobel_x][index_sobel_y]
                gradient_y += pixel_value * sobel_y[index_sobel_x][index_sobel_y]

        magnitude = math.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = min(magnitude, clamped_magnitude)
        destination[x, y] = magnitude

# Perform the GPU Kernel Sobel
def gpu_sobel(source, threads_per_block, blocks_per_grid):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    clamped_magnitude = 175

    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array_like(source)
    device_sobel_x = cuda.to_device(sobel_x)
    device_sobel_y = cuda.to_device(sobel_y)

    sobel_kernel[blocks_per_grid, threads_per_block](device_input_image, device_output_image, device_sobel_x, device_sobel_y, clamped_magnitude)
    return device_output_image.copy_to_host()

# Perform threshold kernel on a single pixel
@cuda.jit
def threshold_kernel(source, destination, low_threshold, high_threshold):
    x, y = cuda.grid(2)

    if x < source.shape[0] and y < source.shape[1]:
        if source[x, y] >= high_threshold:
            destination[x, y] = 255  # Potential edge
        elif source[x, y] >= low_threshold:
            destination[x, y] = 128 # Weak edge 
        else:
            destination[x, y] = 0

# Perform the GPU Kernel Threshold
def gpu_threshold(source, threads_per_block, blocks_per_grid):
    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array_like(source)
    lowThreshold = 51
    highThreshold = 102 

    threshold_kernel[blocks_per_grid, threads_per_block](device_input_image, device_output_image, lowThreshold, highThreshold)

    return device_output_image.copy_to_host()

# Perform hysterisis kernel on a single pixel
@cuda.jit
def hysterisis_kernel(source, destination):
    x, y = cuda.grid(2)
    width, height = source.shape[:2]

    if x < width and y < height:
        if source[x, y] == 255:
            destination[x, y] = 255
        elif source[x, y] == 0:
            destination[x, y] = 0
        else:
            for offset_x in range(-1, 2, 1):
                for offset_y in range(-1, 2, 1):
                    source_x = offset_x + x
                    source_y = offset_y + y

                    if source_x >= 0 and source_y >= 0 and source_x < width and source_y < height:
                        if source[source_x, source_y] == 255:
                            destination[x, y] = 255
                            return

            destination[x, y] = 0


# Perform the GPU Kernel Hysterisis
def gpu_hysterisis(source, threads_per_block, blocks_per_grid):
    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array_like(source)

    hysterisis_kernel[blocks_per_grid, threads_per_block](device_input_image, device_output_image)

    return device_output_image.copy_to_host()

# Get all args
def getAllArgs():
    global input_image, output_image, apply_args, thread_block_size

    parser = argparse.ArgumentParser()

    parser.add_argument("--tb", type=int, help="optional size of a thread block for all operations")
    parser.add_argument("--bw", action="store_true", help="perform only the bw_kernel")
    parser.add_argument("--gauss", action="store_true", help="perform the bw_kernel and the gauss_kernel")
    parser.add_argument("--sobel", action="store_true", help="perform all kernels up to sobel_kernel and write to disk the magnitude of each pixel")
    parser.add_argument("--threshold", action="store_true", help="perform all kernels up to threshold_kernel")
    parser.add_argument("inputImage", type=str, help="the source image")
    parser.add_argument("outputImage", type=str, help="the destination image")

    args = parser.parse_args()

    if args.tb is not None:
        thread_block_size = (args.tb, args.tb)
    if args.bw:
        apply_args[1:] = [False, False, False, False]
    if args.gauss:
        apply_args[2:] = [False, False, False]
    if args.sobel:
        apply_args[3:] = [False, False]
    if args.threshold:
        apply_args[4] = False
    
    if args.inputImage is not None:
        input_image = args.inputImage

    if args.outputImage is not None:
        output_image = args.outputImage

if __name__ == '__main__':
    getAllArgs()
    opened_image = Image.open(input_image)
    np_array_image = np.array(opened_image)
    blocks_per_grid = calcultate_blocks_per_grid(np_array_image)

    # Perform BW Kernel
    if apply_args[0]:
        output_bw = gpu_rgb_to_bw(np_array_image, thread_block_size, blocks_per_grid)
        np_array_image = output_bw

    # Perform Gauss Kernel
    if apply_args[1]:
        output_gauss = gpu_gaussian_blur(np_array_image, thread_block_size, blocks_per_grid)
        np_array_image = output_gauss

    # Perform Sobel Kernel
    if apply_args[2]:
        magnitude = gpu_sobel(np_array_image, thread_block_size, blocks_per_grid)
        # You can use magnitude and angle arrays for further processing or visualization
        np_array_image = magnitude.astype(np.uint8)

    # Perform Threshold Kernel    
    if apply_args[3]:
        thresholded_image = gpu_threshold(np_array_image, thread_block_size, blocks_per_grid)
        np_array_image = thresholded_image

    # Perform Hysterisis Kernel
    if apply_args[4]:
        hysterisis = gpu_hysterisis(np_array_image, thread_block_size, blocks_per_grid)
        np_array_image = hysterisis

    m = Image.fromarray(np_array_image)
    m.save(output_image)
