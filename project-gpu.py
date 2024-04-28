#!/usr/bin/python
import argparse

inputImage = None
outputImage = None
nombreThreads = 16
applyBw = False
applyGaussian = False
applySobel = False
applyThreshold = False

def getAllArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--tb", type=int)
    parser.add_argument("--bw")
    parser.add_argument("--gauss")
    parser.add_argument("--sobel")
    parser.add_argument("--threshold")
    parser.add_argument("inputImage", type=str)
    parser.add_argument("outputImage", type=str)
    
    args = parser.parse_args()

    if (args.tb != None):
        nombreThreads = args.tb
    if (args.bw != None):
        applyBw = True
    if (args.gauss != None):
        applyGaussian = True
    if (args.sobel != None):
        applySobel = True
    if (args.threshold != None):
        applyThreshold = True
    if (args.inputImage != None):
        inputImage = args.inputImage
    if (args.outputImage != None):
        outputImage = args.outputImage

if __name__ == '__main__':
    getAllArgs()
    print(f"{ok}")
    print(f"Input image : {inputImage}\nOutputImage : {outputImage}")# \nNombreThreads : {nombreThreads}\nBW : {applyBw}\nGaussian : {applyGaussian}\nSobel : {applySobel}\nThreshold : {applyThreshold} ")