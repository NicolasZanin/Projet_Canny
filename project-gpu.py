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
    
    if applyBw:
        output = gpu_rgb_to_bw(temptab)
        m = Image.fromarray(output)
        m.save(outputImage)
