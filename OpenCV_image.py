import cv2
import numpy as np
img=cv2.imread('d1.jpg')
if img is None:
    print("Image load failed")
    exit()
def cut(event,x,y,f,p):
    global sx,sy,ex,ey,img
    if event==cv2.EVENT_LBUTTONDOWN:
        sx,sy=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        ex,ey=x,y
        img=img[sy:ey,sx:ex]
    cv2.imshow('img',img)
cv2.namedWindow('img')
cv2.imshow('img',img)
cv2.setMouseCallback('img',cut)
cv2.waitKey()


import numpy as np
old_im=img
gr_old=cv2.cvtColor(old_im,cv2.COLOR_BGR2GRAY)
new_im=cv2.imread('d1.jpg')
gr_new=cv2.cvtColor(new_im,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create()
old_kp,old_des=sift.detectAndCompute(gr_old,None)
new_kp,new_des=sift.detectAndCompute(gr_new,None)
print(len(old_kp),len(new_kp))
flann_matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matcher=flann_matcher.knnMatch(old_des,new_des,2)
T=0.7
# print(knn_matcher[1][0].distance/knn_matcher[1][1].distance<T)
m_l=[]
for old_des,new_des in knn_matcher:
    if old_des.distance/new_des.distance<T:
        m_l.append(old_des)
mc_img=np.empty((new_im.shape[0],old_im.shape[1]+new_im.shape[1],3),np.uint8)
cv2.drawMatches(old_im,old_kp,new_im,new_kp,m_l,mc_img,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("end_im",mc_img)
cv2.waitKey()
cv2.destroyAllWindows()