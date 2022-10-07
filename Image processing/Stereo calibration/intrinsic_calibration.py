import numpy as np
import glob
import sys
import os
import yaml
import cv2 as cv


def intrinsic_calibration(img_folder, pattern_shape=(4, 8), show=True):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_shape[0]*pattern_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_shape[0],0:pattern_shape[1]].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(os.path.join("data", img_folder, '*'))
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_shape, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            if show:
                cv.drawChessboardCorners(img, pattern_shape, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(300)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    img = cv.imread(images[0])
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    if show:
        cv.imshow("Example: distorted image", img)
        cv.imshow("Example: undistorted image", dst)
        cv.waitKey(5000)


    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("Finish calibration")
    print("Mean reprojection error (in pixels): {}".format(mean_error/len(objpoints)) )


    params = {
        "K": newcameramtx.tolist(),
        "dist": dist.tolist(),
        "roi": list(roi)
    }
    with open(os.path.join("results", img_folder+".yml"), 'w') as f:
        yaml.dump(params, f)



if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    try:
        img_folder = sys.argv[1]
        pattern_shape = (int(sys.argv[2]), int(sys.argv[3]))

    except:
        print("invalid arguments...")
        exit()



    intrinsic_calibration(img_folder=img_folder, pattern_shape=pattern_shape)



