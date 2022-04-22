import cv2
import numpy as np
import glob

from DataLoader import *

def main():
    data_path = './Data/*.jpg'
    calib_path = './Data/calibration.txt'
    features_path = './Data/matching*.txt'

    K = readK(calib_path)
    mp1, mp2, img_pairs = getFeatures(features_path)

    