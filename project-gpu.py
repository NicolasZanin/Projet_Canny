#!/usr/bin/env python3
import argparse
import numpy as np
from numba import cuda
import numba as nb
from PIL import Image
import sys
import math
import time
from timeit import default_timer as timer

inputImage = None
outputImage = None
nombreThreads = 16
applyBw = False
applyGaussian = False
applySobel = False
applyThreshold = False
threadBlockSize = (8, 8)  # Default thread block size

# Calculate Blocks par grid necessary for the source image depending on threadBlockSize
def calcultateBlocksPerGrid(imageSource):
    width, height = imageSource.shape[0], imageSource.shape[1]
    blockspergrid_x = math.ceil(width / threadBlockSize[0])
    blockspergrid_y = math.ceil(height / threadBlockSize[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return blockspergrid

# Convert color to gray with GPU
@cuda.jit
def RGBToBWKernel(source, destination, offset):
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
def gpu_rgb_to_bw(source, threadsPerBlock, blockspergrid):
    width, height = source.shape[:2]
    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array((width, height), dtype=np.uint8)

    for offset in range(33, 1, -1):

        # Here is 6 iterations by default, but if we raise this number, that upgrade the image quality in output
        for i in range(6):
            RGBToBWKernel[blockspergrid, threadsPerBlock](device_input_image, device_output_image, offset)
            cuda.synchronize()

    output = device_output_image.copy_to_host()
    return output

# Performs Gaussian blurring kernel on a single pixel
@cuda.jit
def GaussianBlurKernel(source, destination, filter):
    x, y = cuda.grid(2)
    sizeFilterX, sizeFilterY = filter.shape[:2]
    
    if x < source.shape[0] and y < source.shape[1]:
        width, height = source.shape

        # Apply the convolution filter on a single pixel  
        weighted_sum = 0
        normalization_factor = 0
        
        for indexFilterX in range(sizeFilterX):
            for indexFilterY in range(sizeFilterY):
                source_x = x + indexFilterX - sizeFilterX // 2
                source_y = y + indexFilterY - sizeFilterY // 2
                
                if source_x >= 0 and source_x < width and source_y >= 0 and source_y < height:
                    pixel_value = source[source_x, source_y]
                else:
                    pixel_value = source[x, y]
                
                weighted_sum += pixel_value * filter[indexFilterX][indexFilterY]
                normalization_factor += filter[indexFilterX][indexFilterY]
        
        if normalization_factor != 0:
            destination[x, y] = int(math.ceil(weighted_sum // normalization_factor))
        else:
            destination[x, y] = 0

# Perform the GPU Kernel gaussian blur
def gpu_gaussian_blur(image, threadsPerBlock, blocksPerGrid):
    device_input_image = cuda.to_device(image)
    device_output_image = cuda.device_array_like(image)
    filter = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]])

    GaussianBlurKernel[blocksPerGrid, threadsPerBlock](device_input_image, device_output_image, filter)
    return device_output_image.copy_to_host()

# Performs sobel kernel on a single pixel
@cuda.jit
def SobelKernel(source, destination, sobel_x, sobel_y, clamped_magnitude):
    x, y = cuda.grid(2)
    sizeSobelX = sobel_x.shape

    if x < source.shape[0] and y < source.shape[1]:
        # Apply the sobel mask on a single pixel
        width, height = source.shape[:2]
        gradient_x = 0
        gradient_y = 0
        
        for indexSobelX in range(sizeSobelX[0]):
            for indexSobelY in range(sizeSobelX[1]):
                source_x = x + indexSobelX - sizeSobelX[0] // 2
                source_y = y + indexSobelY - sizeSobelX[1] // 2
                
                if source_x >= 0 and source_x < width and source_y >= 0 and source_y < height:
                    pixel_value = source[source_x, source_y]
                else:
                    pixel_value = source[x, y]  
                
                gradient_x += pixel_value * sobel_x[indexSobelX][indexSobelY]
                gradient_y += pixel_value * sobel_y[indexSobelX][indexSobelY]
        
        magnitude = math.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = min(magnitude, clamped_magnitude)
        destination[x, y] = magnitude

# Perform the GPU Kernel Sobel
def gpu_sobel(source, threadsPerBlock, blocksPerGrid):
    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array_like(source)
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    clamped_magnitude = 175

    SobelKernel[blocksPerGrid, threadsPerBlock](device_input_image, device_output_image, sobel_x, sobel_y, clamped_magnitude)
    return device_output_image.copy_to_host()

# Performs threshold kernel on a single pixel
@cuda.jit
def ThresholdKernel(source, destination, lowThreshold, highThreshold):
    x, y = cuda.grid(2)
    
    if x < source.shape[0] and y < source.shape[1]:
        if source[x, y] >= highThreshold:
            destination[x, y] = 255  # Potential edge
        elif source[x, y] >= lowThreshold:
            destination[x, y] = 128 # Weak edge 
        else:
            destination[x, y] = 0

# Perform the GPU Kernel Threshold
def gpu_threshold(source, threadsPerBlock, blocksPerGrid):
    device_input_image = cuda.to_device(source)
    device_output_image = cuda.device_array_like(source)
    lowThreshold = 51
    highThreshold = 102 

    ThresholdKernel[blocksPerGrid, threadsPerBlock](device_input_image, device_output_image, lowThreshold, highThreshold)

    return device_output_image.copy_to_host()

def getAllArgs():
    global inputImage, outputImage, nombreThreads, applyBw, applyGaussian, applySobel, applyThreshold, threadBlockSize
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--tb", type=int)
    parser.add_argument("--bw", action="store_true")
    parser.add_argument("--gauss", action="store_true")
    parser.add_argument("--sobel", action="store_true")
    parser.add_argument("--threshold", action="store_true")
    parser.add_argument("inputImage", type=str)
    parser.add_argument("outputImage", type=str)
    
    args = parser.parse_args()

    if args.tb is not None:
        threadBlockSize = (args.tb, args.tb)
    else:
        threadBlockSize = (8, 8)  # Default thread block size

    if args.bw:
        applyBw = True
    if args.gauss:
        applyBw = True
        applyGaussian = True
    if args.sobel:
        applyBw = True
        applyGaussian = True
        applySobel = True
    if args.threshold:
        applyBw = True
        applyGaussian = True
        applySobel = True
        applyThreshold = True
    if args.inputImage is not None:
        inputImage = args.inputImage
    if args.outputImage is not None:
        outputImage = args.outputImage

if __name__ == '__main__':
    getAllArgs()
    print(f"Input image : {inputImage}\nOutputImage : {outputImage}")
    
    openedImage = Image.open(inputImage)
    npArrayImage = np.array(openedImage)
    blocksPerGrid = calcultateBlocksPerGrid(npArrayImage)
    
    if applyBw:         
        output_bw = gpu_rgb_to_bw(npArrayImage, threadBlockSize, blocksPerGrid)
        npArrayImage = output_bw

    if applyGaussian:
        output_gauss = gpu_gaussian_blur(npArrayImage, threadBlockSize, blocksPerGrid)
        npArrayImage = output_gauss

    if applySobel:
        magnitude = gpu_sobel(npArrayImage, threadBlockSize, blocksPerGrid)
        # You can use magnitude and angle arrays for further processing or visualization
        npArrayImage = magnitude.astype(np.uint8)
            
    if applyThreshold:
        thresholded_image = gpu_threshold(npArrayImage, threadBlockSize, blocksPerGrid)
        npArrayImage = thresholded_image

    m = Image.fromarray(npArrayImage)
    m.save(outputImage)
