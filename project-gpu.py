#!/usr/bin/env python3
import argparse
import numpy as np
from numba import cuda
import numba as nb
from PIL import Image
import sys
import math

inputImage = None
outputImage = None
nombreThreads = 16
applyBw = False
applyGaussian = False
applySobel = False
applyThreshold = False

@cuda.jit
def RGBToBWKernel(source, destination, offset):
    height = source.shape[1]
    width = source.shape[0]
    x,y = cuda.grid(2)
    if (x<width and y<height) :
        r_x= (x+offset)%width
        r_y= (y+offset)%height
        destination[r_x,r_y]=np.uint8(0.3*source[r_x,r_y,0]+0.59*source[r_x,r_y,1]+0.11*source[r_x,r_y,2])

def gpu_rgb_to_bw(image):
    threadsperblock = (8, 8)
    width, height = image.shape[1], image.shape[0]
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    s_image = cuda.to_device(image)
    d_image = cuda.device_array((image.shape[0], image.shape[1]), dtype=np.uint8)

    offset_range = range(33, 1, -1)
    for off in offset_range:
        runs = 6
        result = np.zeros(runs, dtype=np.float32)
        for i in range(runs):
            RGBToBWKernel[blockspergrid, threadsperblock](s_image, d_image, off)
            cuda.synchronize()

        output = d_image.copy_to_host()

    return output

@cuda.jit
def GaussianBlurKernel(source, destination, filter):
    x, y = cuda.grid(2)
    if x < source.shape[1] and y < source.shape[0]:
        height, width, channel = source.shape
        for c in range(channel):
            weighted_sum = 0
            normalization_factor = 0
            for i in range(len(filter)):
                for j in range(len(filter[0])):
                    nx = x + i - len(filter) // 2
                    ny = y + j - len(filter[0]) // 2
                    if nx >= 0 and nx < width and ny >= 0 and ny < height:
                        pixel_value = source[ny, nx, c]
                    else:
                        pixel_value = source[y, x, c]  
                    weighted_sum += pixel_value * filter[i][j]
                    normalization_factor += filter[i][j]
            if normalization_factor != 0:
                destination[y, x, c] = int(weighted_sum / normalization_factor)
            else:
                destination[y, x, c] = source[y, x, c]  

@cuda.jit
def GradientsKernel(source, destination, sobel_x, sobel_y, max_value):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        height, width = source.shape[:2]
        gradient_x = 0
        gradient_y = 0
        for i in range(len(sobel_x)):
            for j in range(len(sobel_x[0])):
                nx = x + i - len(sobel_x) // 2
                ny = y + j - len(sobel_x[0]) // 2
                if nx >= 0 and nx < width and ny >= 0 and ny < height:
                    pixel_value = source[ny, nx]
                else:
                    pixel_value = source[y, x]  
                gradient_x += pixel_value * sobel_x[i][j]
                gradient_y += pixel_value * sobel_y[i][j]
        magnitude = math.sqrt(gradient_x**2 + gradient_y**2)
        magnitude = min(magnitude, max_value)  # Clamp the magnitude
        destination[y, x] = magnitude

@cuda.jit
def ThresholdKernel(source, destination, threshold):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        if source[y, x] >= threshold:
            destination[y, x] = 255  # Potential edge
        else:
            destination[y, x] = 0

def gpu_gaussian_blur(image, threadsPerBlock, blocksPerGrid):
    s_image = cuda.to_device(image)
    d_image = cuda.device_array_like(image)
    filter = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]])

    GaussianBlurKernel[blocksPerGrid, threadsPerBlock](s_image, d_image, filter)

    return d_image.copy_to_host()

def compute_gradients(image, threadsPerBlock, blocksPerGrid):
    s_image = cuda.to_device(image)
    d_image = cuda.device_array_like(image)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    max_value = 175
    GradientsKernel[blocksPerGrid, threadsPerBlock] (s_image, d_image, sobel_x, sobel_y, max_value)
    return d_image.copy_to_host()

def threshold_image(image, threadsPerBlock, blocksPerGrid, threshold):
    s_image = cuda.to_device(image)
    d_image = cuda.device_array_like(image)

    ThresholdKernel[blocksPerGrid, threadsPerBlock](s_image, d_image, threshold)

    return d_image.copy_to_host()

def getAllArgs():
    global inputImage, outputImage, nombreThreads, applyBw, applyGaussian, applySobel, applyThreshold
    
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
        nombreThreads = args.tb
    if args.bw:
        applyBw = True
    if args.gauss:
        applyGaussian = True
    if args.sobel:
        applySobel = True
    if args.threshold:
        applyThreshold = True
    if args.inputImage is not None:
        inputImage = args.inputImage
    if args.outputImage is not None:
        outputImage = args.outputImage

if __name__ == '__main__':
    getAllArgs()
    print("ok")
    print(f"Input image : {inputImage}\nOutputImage : {outputImage}")
    temp = Image.open(inputImage)
    temptab = np.array(temp)
    
    if applyGaussian:  
        threadsPerBlock = (8, 8)
        width, height = temptab.shape[1], temptab.shape[0]
        blocksPerGrid_x = math.ceil(width / threadsPerBlock[0])
        blocksPerGrid_y = math.ceil(height / threadsPerBlock[1])
        blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)

        output_gauss = gpu_gaussian_blur(temptab, threadsPerBlock, blocksPerGrid)
        temptab = output_gauss

    if applyBw:         
        output_bw = gpu_rgb_to_bw(temptab)
        temptab = output_bw

    if applySobel:
        if applyBw:
            threadsPerBlock = (8, 8)
            width, height = temptab.shape[1], temptab.shape[0]
            blocksPerGrid_x = math.ceil(width / threadsPerBlock[0])
            blocksPerGrid_y = math.ceil(height / threadsPerBlock[1])
            blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
          
            magnitude = compute_gradients(temptab, threadsPerBlock, blocksPerGrid)
            # You can use magnitude and angle arrays for further processing or visualization
            temptab = magnitude.astype(np.uint8)
            
            if applyThreshold:
                threshold_value = 100  # Adjust threshold value as needed
                thresholded_image = threshold_image(temptab, threadsPerBlock, blocksPerGrid, threshold_value)
                temptab = thresholded_image

    m = Image.fromarray(temptab)
    m.save(outputImage)
