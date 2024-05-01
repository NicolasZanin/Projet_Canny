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
