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
def RGBToBWKernel(source, destination, offset):
    height = source.shape[1]
    width = source.shape[0]
    #offset =8 
    x,y = cuda.grid(2)
    if (x<width and y<height) :
        r_x= (x+offset)%width
        r_y= (y+offset)%height
        # ( (0.3 * R) + (0.59 * G) + (0.11 * B) )
        destination[r_x,r_y]=np.uint8(0.3*source[r_x,r_y,0]+0.59*source[r_x,r_y,1]+0.11*source[r_x,r_y,2])



def gpu_run(imagetab,threadsperblock, blockspergrid ):
        print("Sending image to device ", end=" ")
        #start = timer()
        s_image = cuda.to_device(imagetab)
        d_image = cuda.device_array((imagetab.shape[0],imagetab.shape[1],3),dtype = np.uint8)
        
        
        for off in range(33,1,-1):
            print("--- offset ---", off)
            runs = 6
            result =np.zeros(runs, dtype=np.float32)
            for i in range(runs):
                print("Executing kernel  ", end=" ")
                start = timer()
                RGBToBWKernel[blockspergrid, threadsperblock](s_image, d_image,off) 
                cuda.synchronize()
                dt = timer() - start
                print(" ", dt, " s")
                result[i]=dt
                
            # dt = timer() - start
            #print(" ", dt, " s")
            print("Average :", threadsperblock[0],off, np.average(result[1:]))
        
        output=d_image.copy_to_host()
        return output
        
def cpu_run(source):
    output=np.empty_like(source)
    height = source.shape[1]
    width = source.shape[0]
    print("Executing on CPU   ", end=" ")
    start = timer()
    for x in range(width):
        for y in range(height):
            output[x,y]=np.uint8(0.3*source[x,y,0]+0.59*source[x,y,1]+0.11*source[x,y,2])
    dt = timer() - start
    print(" ", dt, " s")        
    return output
            
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
    
    
if len(sys.argv) < 3:
    print("Usage: ", sys.argv[0]," <inputFile> <outputFile>")
    quit(-1)
    
inputFile = sys.argv[1]
outputFile=sys.argv[2]


im = Image.open(inputFile)
imagetab = np.array(im)

threadsperblock,blockspergrid=compute_threads_and_blocks(imagetab)
output=gpu_run(imagetab, threadsperblock, blockspergrid)
# output=cpu_run(imagetab)
m = Image.fromarray(output) #.convert('RGB')
m.save(outputFile)