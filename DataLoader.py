import numpy as np
import glob

def readK(filedir):
    calibstr = open(filedir, 'r')
    calibstr = calibstr.read()
    calibstr = calibstr.split(' ')
    camMatrix = np.zeros((3,3))
    i,j=0,0
    for string in calibstr[2:]:
        if len(string) > 0:
            string = string.replace("\n", "").replace("[", "").replace("]", "").replace(";", "")
            num = float(string)
            camMatrix[i,j] = num
            j += 1
            if j == 3:
                j = 0
                i += 1
    return camMatrix

def loadFeatures(filepath):
    featurefiles = glob.glob(filepath)
    featurefiles.sort()

    features_x = []
    features_y = []
    feature_matching_map = []

    for i in range(len(featurefiles)):
        features = open(featurefiles[i], 'r')
        features = features.readlines()

        for j in range(1, len(features)): # skip first line
            line = features[j].split(' ')
            matches = int(line[0]) - 1
            match_x = np.zeros((6,))
            match_y = np.zeros((6,))
            feature_matching_line = np.zeros((6,),dtype = int)
            feature_matching_line[i] = 1
            match_x[i] = line[4]
            match_y[i] = line[5]

            line = line[6:]
            for _ in range(matches):
                match_img = int(line[0]) - 1
                feature_matching_line[match_img] = 1
                match_x[match_img] = line[1]
                match_y[match_img] = line[2]
                line = line[3:]
            features_x.append(match_x)
            features_y.append(match_y)
            feature_matching_map.append(feature_matching_line)

    return np.array(features_x), np.array(features_y), np.array(feature_matching_map), len(featurefiles)+1
