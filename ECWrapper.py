import cv2
import numpy as np
import glob

# code for camera calibration and undistortion: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# feature matching: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

def calibrate(save_vals=True):

    video_path = 'checkerboard.mov'

    # Define the dimensions of checkerboard
    CHECKERBOARD = (11, 8)  # Metal Calibration Board

    # stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # get fps of a video
    vidcap = cv2.VideoCapture(video_path)
    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    i = 0

    while vidcap.grab():
        success, image = vidcap.read()

        # check success or not
        if not success:
            break

        if i%10==0: # only take 3 frames per second
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            print("Finding Corners")
            ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
            # If desired number of corners can be detected then, refine the pixel coordinates and display them on the images of checker board
            if ret == True:
                print("Refining pixel coordinates and display on checkerboard")
                threedpoints.append(objectp3d)
        
                # Refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
        
                twodpoints.append(corners2)
        
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
        
                cv2.imshow('img', image)
                cv2.waitKey(1)
        i+=1
    cv2.destroyAllWindows()
    
    print("Taking total of ", len(twodpoints), " frames to calibrate out of ", i, " frames.")

    # Perform camera calibration by passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the detected corners (twodpoints)
    print("Calibrating...")
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
    
    # Check the reprojection error of the found parameters
    mean_error = 0
    for i in range(len(threedpoints)):
        twodpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
        error = cv2.norm(twodpoints[i], twodpoints2, cv2.NORM_L2)/len(twodpoints2)
        mean_error += error
    error = mean_error/len(threedpoints)
    print( "total error: {}".format(error) )
    
    if save_vals:
        # Save the camera calibration result for later use
        outfile = open("cam_intrinsic.txt", "w+")
        outfile.write('Camera matrix\n')
        outfile.write(str(matrix) + '\n')
        outfile.write('Distortion coefficient\n')
        outfile.write(str(distortion) + '\n')
        outfile.write('Reprojection Error\n')
        outfile.write(str(error) + '\n')
        outfile.close()
    return matrix, distortion

def undistort(images, K, distortion):
    new_images = []
    new_heights = []
    new_widths = []
    for img in images:
        h, w = img.shape[:2]
        newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), 0, (w, h))
        # undistort
        dst = cv2.undistort(img, K, distortion, None, newK)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        new_heights.append(y+h)
        new_widths.append(x+w)
        cv2.imshow("pre distortion", img)
        cv2.imshow("post distortion", dst)
        cv2.waitKey(0)
        new_images.append(dst)
    final_width = min(new_widths)
    final_height = min(new_heights)
    for i in range(len(new_images)):
        new_images[i] = cv2.resize(new_images[i], (final_width, final_height))
        cv2.imwrite('./set1/undistorted_'+str(i)+'.jpg', new_images[i])
    return new_images

def readImages(filepath):
    imagefiles = glob.glob(filepath)
    imagefiles.sort()
    images = []
    for i in range(len(imagefiles)):
        images.append(cv2.imread(imagefiles[i]))
    return images

def feature_matching(images):
    dist_thres = 0.1
    features = {}
    for i in range(1,len(images)):
        features[(i)] = {}
        for j in range(0,i):
            features[(i)][(j)] = [] # indexes of features
            img1 = images[i]
            img2 = images[j]
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            print('Finding matches...')
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            print(len(matches))
            good = []
            for m,n in matches:
                curr_dist = m.distance/n.distance
                if curr_dist < dist_thres:
                    good.append(m)
                    print(m.queryIdx)
                    # save the indexes
                    # f1_c, f1_r = f1_indexes[m.queryIdx]
                    # f2_c, f2_r = f2_indexes[m.trainIdx]
                    # indexes = [f1_c, f1_r, f2_c, f2_r] 
                    # fp_indexes.append(indexes)
                    ## need to add ransac
            break
        break

def main():
    load_intrinsics = True
    use_undistorted = True
    if load_intrinsics:
        # Load the camera calibration result
        infile = open("cam_intrinsic.txt", "r")
        lines = infile.readlines()
        infile.close()
        K = np.zeros((3,3))
        for i in range(len(lines[1:4])):
            l = lines[i+1].replace('[', '').replace(']', '').strip().split(' ')
            for j in range(len(l)):
                K[i,j] = float(l[j])

        distortion = np.zeros(5)
        idx = 0
        for i in range(len(lines[5:7])):
            l = lines[i+5].replace('[', '').replace(']', '').strip().split(' ')
            for j in range(len(l)):
                distortion[idx] = float(l[j])
                idx += 1
        print(distortion)
        # distortion = eval(lines[2])
    else:
        K, distortion = calibrate()

    print("K: \n", K)
    print("\nD: \n", distortion)

    if not use_undistorted:
        img_path = './set1/*.JPG'
        images = readImages(img_path)
        # image = cv2.imread('out.jpg')
        # images = [image]
        undistorted_images = undistort(images, K, distortion)
    else:
        print('feature matching')
        img_path = './set1/*.JPG'
        images = readImages(img_path)
        print('num images', len(images))
        undistorted_images = images ## to use undistorted images, change the line above to take './set1/undistorted_*.JPG'
        feature_matching(undistorted_images)
    
    return

if __name__ == "__main__":
    main()
