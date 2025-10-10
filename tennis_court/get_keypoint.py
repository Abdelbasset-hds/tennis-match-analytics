import cv2
import numpy as np

keypoints = []
def recover_courtpoint(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN :
        keypoints.append([x,y])
        print(f"point {len(keypoints)+1} : ({x}:{y})")
        cv2.circle(img,(x,y),2,(0,0,255),-1)
        
cv2.namedWindow(winname='get_keypoint')
cv2.setMouseCallback("get_keypoint",recover_courtpoint)
img = cv2.imread("data/tennis_court.jpg")
img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
while True :
    cv2.imshow('get_keypoint',img)
    k = cv2.waitKey(1) & 0xFF
    if k ==27 :
        break
cv2.destroyAllWindows
np.save("tracker_stubs/dst_key_point.npy",keypoints)