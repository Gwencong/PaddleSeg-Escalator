import cv2
import numpy as np

def get_contour_approx(pred,img):
    h,w = pred.shape[:2]
    approxs = []
    for i in range(3):
        mask = np.where(pred==i,0,255).astype(np.uint8)
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(contour) for contour in contours]
        indexes = [j for j,area in enumerate(areas) if 0.01<area/(h*w)<0.8]
        contour = contours[indexes[0]]
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        # approx = cv2.convexHull(contour)
        approxs.append(approx)
        cv2.drawContours(img,[approx],-1,(0,255,255))
    cv2.imwrite('test.jpg',img)
    return approxs


