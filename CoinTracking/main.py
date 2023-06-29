import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('tray1.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

kernel = np.ones((3,3),np.float32)/25
dst = cv.filter2D(img,-2,kernel)
#blur=cv.medianBlur(img,7)
edges = cv.Canny(img,100,200)
#blur = cv.bilateralFilter(edges,100,90,80)
#kernel = np.ones((3,3),np.uint8)
countInside=0
countOutside=0
moneyInside=0
moneyOutside=0

edges = cv.Canny(img,50,150,apertureSize = 3)
lines = cv.HoughLinesP(edges,1,np.pi/180,90, minLineLength=50, maxLineGap=5)
#print(lines.shape)
maxx=[]
maxy=[]
maxz=[]

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    maxx.append(x1)
    maxx.append(x2)
    maxy.append(y1)
    maxy.append(y2)

circles = cv.HoughCircles(dst, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=40)
for i in circles[0, :]:
    maxz.append(float(i[2]))

highest=maxz.pop(maxz.index(max(maxz)))
lower=maxz.pop(maxz.index(max(maxz)))

for i in circles[0, :]:
            #  outter circle
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (238, 130, 238), 2)
            # inner circle
    cv.circle(img, (int(i[0]), int(i[1])), 2, (60, 60, 60), 3)
    if(int(i[0])<max(maxx) and int(i[0])>min(maxx) and int(i[1])>min(maxy) and int(i[1])<max(maxy)):
        countInside+=1
        if(float(i[2])<lower):
            moneyInside+=5
        else:
            moneyInside+=500
    else:
        countOutside+=1
        if (float(i[2]) < lower):
            moneyOutside += 5
        else:
            moneyOutside += 500
print(f'liczba monet na tacy {countInside} a poza nią {countOutside}')
print(f'liczba pieniedzy na tacy {moneyInside} gr a poza nią {moneyOutside} gr')
cv.imshow('detected circles', img)
cv.waitKey(0)
cv.destroyAllWindows()
