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

def getFeatures(filepath):
    featurefiles = glob.glob(filepath)
    featurefiles.sort() 
    mp1 = []
    mp2 = []
    # rgb = []
    img_pairs = []
    for featurefile in featurefiles:
        main_img = featurefile[-5]
        print(main_img)
        featurestr = open(featurefile, 'r')
        featurestr = featurestr.readlines()

        for i in range(len(featurestr)):
            if i == 0:
                line = featurestr[i].split(':')
                numFeatures = int(line[1])
            else:
                line = featurestr[i].split(' ')
                numFeatures = int(line[0])
                # featurergb = np.array([int(line[1]), int(line[2]), int(line[3])])
                mainFeature = [float(line[4]), float(line[5]), float(1)]
                j = 6
                while numFeatures>1:
                    # rgb.append(featurergb)
                    pair = [main_img,line[j]]
                    img_pairs.append(pair)
                    mp1.append(mainFeature)
                    mp2.append([float(line[j+1]), float(line[j+2]), float(1)])
                    j += 3
                    numFeatures -= 1
    return np.array(mp1), np.array(mp2), np.array(img_pairs)
