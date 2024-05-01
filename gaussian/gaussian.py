#!/usr/bin/env python3

from numba import cuda
import numba as nb
import numpy as np
import time
from PIL import Image
from timeit import default_timer as timer
import math
import sys

@cuda.jit
def GaussianBlur(source, destination, filter):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
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
                        pixel_value = source[y, x, c]  # Use current pixel value as substitute
                    weighted_sum += pixel_value * filter[i][j]
                    normalization_factor += filter[i][j]
            # Check if normalization factor is non-zero to avoid division by zero
            if normalization_factor != 0:
                destination[y, x, c] = int(weighted_sum / normalization_factor)
            else:
                destination[y, x, c] = source[y, x, c]  # Preserve original pixel value for edge pixels



    

def compute_threads_and_blocks(imagetab):
    threadsperblock = (8,8)
    if len(sys.argv) ==4:
        threadsperblock=(int(sys.argv[3]),int(sys.argv[3]))
    width, height = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print("Thread blocks ", threadsperblock)
    print("Grid ", blockspergrid)
    return threadsperblock,blockspergrid

def cpu_run(source, filtre):
    output = np.empty_like(source)
    height, width, channel = source.shape
    print("Executing on CPU   ", end=" ")
    start = timer()
    for c in range(channel): 
        for x in range(width): 
            for y in range(height): 
                # Compute weighted sum using Gaussian kernel
                weighted_sum = 0
                normalization_factor = 0
                for i in range(len(filtre)):
                    for j in range(len(filtre[0])):
                        # Coordinates of the neighboring pixel
                        nx = x + i - len(filtre) // 2
                        ny = y + j - len(filtre[0]) // 2
                        
                        # Handling edges
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            # Use current pixel value as substitute
                            pixel_value = source[y, x, c]
                        else:
                            pixel_value = source[ny, nx, c]
                        
                        weighted_sum += pixel_value * filtre[i][j]
                        normalization_factor += filtre[i][j]
                
                # Normalize and store the result
                output[y, x, c] = int(weighted_sum / normalization_factor)
    
    end = timer()
    print("Execution time: {:.6f} seconds".format(end - start))
    return output

def gpu_run(imagetab, threadsPerBlock, blocksPerGrid):
    print("sending images to device ", end=" ")
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array_like(imagetab)

    filter = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])

    GaussianBlur[blocksPerGrid, threadsPerBlock](s_image, d_image, filter)

    output = d_image.copy_to_host()
    return output



inputFile = sys.argv[1]
outputFile=sys.argv[2]


im = Image.open(inputFile)
imagetab = np.array(im)
filtre = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]

threadsperblock,blockspergrid=compute_threads_and_blocks(imagetab)
output = gpu_run(imagetab, threadsperblock, blockspergrid)
#output=cpu_run(imagetab, filtre)
print(output)
m = Image.fromarray(output) 
m.save(outputFile)