import numpy as np
import cv2 as cv
#btt
def contour(img):
    ss = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    cntr = cv.inRange(ss, lower, upper)
    flou = cv.blur(cntr, (2,2))
    resultas = cv.threshold(flou,0,255,cv.THRESH_BINARY)
    return resultas

def cntr(mask_img):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    h1 = cv.convexh1(contours)
    return contours, h1

def getdfc(contours):
    h1 = cv.convexh1(contours, returnPoints=False)
    dfc = cv.convexitydfc(contours, h1)
    return dfc

cap = cv.VideoCapture("/") #add you path here
while cap.isOpened():
    _, img = cap.read()
    try:
        mask_img = contour(img)
        contours, h1 = cntr(mask_img)
        cv.drawContours(img, [contours], -1, (255,255,0), 2)
        cv.drawContours(img, [h1], -1, (0, 255, 255), 2)
        dfc = getdfc(contours)
        if dfc is not None:
            cnt = 0
            for i in range(dfc.shape[0]):  
                s, e, f, d = dfc[i][0]
                begin = tuple(contours[s][0])
                end = tuple(contours[e][0])
                tut = tuple(contours[f][0])
                x = np.sqrt((end[0] - begin[0]) ** 2 + (end[1] - begin[1]) ** 2)
                y = np.sqrt((tut[0] - begin[0]) ** 2 + (tut[1] - begin[1]) ** 2)
                z = np.sqrt((end[0] - tut[0]) ** 2 + (end[1] - tut[1]) ** 2)
                angle = np.arccos((y ** 2 + z ** 2 - x ** 2) / (2 * y * z))  
                if angle <= np.pi / 2:  
                    cnt += 1
                    cv.circle(img, tut, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt+1
            cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
        cv.imshow("img", img)
    except:
        pass
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
