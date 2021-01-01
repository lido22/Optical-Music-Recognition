from commonfunctions import *
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from collections import Counter
import skimage.filters as fr
import skimage as sk
import cv2, time
from skimage.transform import hough_line, hough_line_peaks, rotate


def binraization(img,n=8,t=15):

    outputimg = np.zeros(img.shape)
    intimg = np.zeros(img.shape)
    h = img.shape[1]
    w = img.shape[0]
    s= min(w,h)//n
    count = s**2
    img = np.pad(img,s,"constant")
    intimg = np.cumsum(img ,axis =1)
    intimg = np.cumsum(intimg ,axis =0)
    a = np.roll(intimg,-s//2,axis =0)
    a = np.roll(a,-s//2,axis =1)
    a[:,-s//2:]=a[-s//2-1,-s//2-1]
    a[-s//2:,:]=a[-s//2-1,-s//2-1]
    b = np.roll(intimg,s//2+1,axis =0)
    b = np.roll(b,-s//2,axis =1)
    b[0:s//2+1,:]=0
    b[:,-s//2:]=0
    
    c = np.roll(intimg,s//2+1,axis =1)
    c = np.roll(c,-s//2,axis =0)
    c[:,0:s//2+1]=0
    c[-s//2:,:]=0
    
    d = np.roll(intimg,s//2+1,axis =0)
    d = np.roll(d,s//2+1,axis =1)
    d[0:s//2+1,:]=0
    d[:,0:s//2+1]=0

    sum = (a-b-c+d)*(100-t)/100
    outputimg = (img>sum/count)*255
    return outputimg[s:-s,s:-s]



def getRefLengths(img):
    cols = img.shape[1]
    rows = img.shape[0]
    hist = np.zeros((rows,rows), dtype=np.uint32)
    
    for i in range(0, cols):
        a = img[:,i]
        starts = np.array((a[:-1] == 1) & (a[1:] == 0))
        starts_ix = np.where(starts)[0] + 2
        ends = np.array((a[:-1] == 0) & (a[1:] == 1))
        ends_ix = np.where(ends)[0] + 2
        s1 = starts_ix.size
        s2 = ends_ix.size
        if s2 > s1:
            starts_ix = np.pad(starts_ix,(s2-s1,0), mode='constant', constant_values=(1))
        elif s1 > s2:
            ends_ix = np.pad(ends_ix,(0,s1-s2), mode='constant', constant_values=(a.size + 1))
        elif s1 > 0 and s2 > 0 and starts_ix[0] > ends_ix[0]:
            starts_ix = np.pad(starts_ix,(1,0), mode='constant', constant_values=(1))
            ends_ix = np.pad(ends_ix,(0,1), mode='constant', constant_values=(a.size + 1))
            
        l0 = ends_ix - starts_ix
        starts_ix1 = np.pad(starts_ix[1:],(0,1), mode='constant', constant_values=(a.size + 1)) 
        l1 = starts_ix1 - (starts_ix + l0)
        for i in range(s1):
            hist[l0[i], l1[i]] += 1
       
    hist[:,0] = 0
    mx = np.max(hist)
    ind = np.where(hist == mx)
    return ind[0][0], ind[1][0]
  
  
def deleteLines(binary,w):
    theta = np.arange(-math.pi,math.pi,0.01)
    max_R = math.sqrt(binary.shape[0]**2 +  binary.shape[1]**2)
    vote_mat = np.zeros((int(theta.size),int(round(max_R))))
    for row in range(binary.shape[0]):
        for col in range(binary.shape[1]):
             if binary[row,col]==1:
                for i in range(theta.size):
                    R = math.cos(theta[i]) * col + math.sin(theta[i]) * row
                    if  R < int(round(max_R)) and R >= 0:
                        vote_mat[i,int(R)]+=1
    for j in range(vote_mat.shape[0]):
        for k in range(vote_mat.shape[1]):
            if vote_mat[j,k] > 100:
                thval = theta[j]
                rval = k
                for i in range(binary.shape[1]):
                    x1 = i
                    y1 = int(round((rval - math.cos(thval)*x1)/ math.sin(thval)))
                    if not binary[y1-w:y1,x1].all() and not binary[y1:y1+w,x1].all():
                        binary[y1,x1] = 0
    show_images([binary],['line removed'])
    return binary


def getCandidateStaffs(binaryImg, staffHeight):
    filteredImg = np.copy(binaryImg)
    candidates = [] # Contains list of candidate staffs (row, begin, height)
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
    cur = 0
    upperLimitHeight = staffHeight+2
    lowerLimitHeight = abs(staffHeight-2)
    flag = False
    for i in range(cols):
        for j in range(rows):
            if filteredImg[j,i] == 0 and flag == False:
                beg = j
                flag = True
            elif filteredImg[j,i] == 1 and flag == True:
                flag = False
                if j-beg > upperLimitHeight or j-beg < lowerLimitHeight:
                    filteredImg[beg:j, i] = 1
                else:
                    candidates.append((i, beg, j-beg))
    return filteredImg, candidates


def RemoveThinStaffs(v, filteredImg, staffHeight):
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
    candidates = v.copy()
    img = np.copy(filteredImg)
    eliminated = []
    for i in candidates:
        col, begin, length = i
        maxWidthLeft = 0
        maxWidthRight = 0
        wLeft = 0
        wRight = 0
        while col+wLeft+1 < cols and begin < i[1]+length:
#             print('testing filteredImg[',begin, col+wLeft+1)
            if filteredImg[begin, col+wLeft+1] == 0:
                wLeft += 1
            else:
                begin += 1
                maxWidthLeft = max(maxWidthLeft, wLeft)
                wLeft = 0
        maxWidthLeft = max(maxWidthLeft, wLeft)
        begin = i[1]
        while col-wRight-1 >= 0 and begin < i[1]+length:
#             print('testing filteredImg[',begin, col-wRight-1)
            if filteredImg[begin, col-wRight-1] == 0:
                wRight += 1
#                 print('zero')
            else:
#                 print('one')
                begin += 1
                maxWidthRight = max(maxWidthRight, wRight)
                wRight = 0
        maxWidthRight = max(maxWidthRight, wRight)
                
        width = maxWidthRight + maxWidthLeft +1
        if(width < 2*staffHeight):
            img[i[1]:i[1]+i[2], i[0]] = 1
            v.remove(i)
            eliminated.append(i)
    return img, v, eliminated



def removeLonelyStaffs(v, filteredImg, staffHeight, spaceHeight, eliminated):
    verConnected = False
    horConnected = False
    c = v.copy()
    img = np.copy(filteredImg)
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
    staffs = np.ones(img.shape)

    for i in c:
        verConnected = False
        horConnected = False
        
        col, begin, length = i
        
        if col-1>=0 and np.any(filteredImg[begin:begin+length , col-1] == 0):
            horConnected = True
        if col+1<cols and np.any(filteredImg[begin:begin+length ,col+1] == 0):
            horConnected = True
            
        start = max(begin-(spaceHeight+staffHeight), 0)
        end = min(rows, begin+length+spaceHeight+staffHeight-1)
        if np.any(filteredImg[start:begin, col] == 0) or np.any(filteredImg[begin+length:end, col] == 0):
            verConnected = True
            
        if not (verConnected and horConnected):
            img[i[1]:i[1]+i[2], i[0]] = 1
            v.remove(i)
            eliminated.append(i)
    for i in v:
        staffs[i[1]:i[1]+i[2],i[0]] = 0
    return staffs, v, eliminated


def addFalseNegatives(v, filteredImg, staffHeight, spaceHeight, eliminated):
    verConnected = False
    horConnected = False
    c = v.copy()
    eliminated.copy()
    img = np.copy(filteredImg)
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
    
    for i in eliminated:
        verConnected = False
        horConnected = False
        
        col, begin, length = i
        
        if col-1>=0 and np.any(filteredImg[begin:begin+length , col-1]):
            horConnected = True
        if col+1<cols and np.any(filteredImg[begin:begin+length ,col+1]):
            horConnected = True
        start = max(begin-(spaceHeight+staffHeight), 0)
        end = min(rows, begin+length+spaceHeight+staffHeight)
        if np.any(filteredImg[start:begin, col]) or np.any(filteredImg[begin+length:end, col]):
            verConnected = True
            
        if verConnected and horConnected:
            img[i[1]:i[1]+i[2], i[0]] = 0
            v.append(i)
            
    return img, v


def getStaffLines(binary):
    staffHeight, spaceHeight = getRefLengths(binary)
    print('staff height =', staffHeight, 'space Height =', spaceHeight)    
    filteredImg, candidates = getCandidateStaffs(binary, staffHeight)
    filteredImg1, candidates, eliminated = RemoveThinStaffs(candidates, filteredImg, staffHeight)
    filteredImg2, candidates, eliminated = removeLonelyStaffs(candidates, binary, staffHeight, spaceHeight, eliminated)
    filteredImg3, candidates = addFalseNegatives(candidates, filteredImg2, staffHeight, staffHeight, eliminated)
    return filteredImg3


def getPts(pts, shape):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
#     width = np.linalg.norm(rect[0]-rect[1])
#     height = np.linalg.norm(rect[1]-rect[2])
#     print(rect)
# #     print(shape)
#     const = min(shape[1],shape[0])/15
    xConst = 0
    yConst = 0
    
    if shape[1] > 4000:
        yConst = 250
    elif shape[1] > 3500:
        yConst = 250
    elif shape[1] > 3000:
        yConst = 200
    elif shape[1] > 2500:
        yConst = 100
    elif shape[1] > 2000:
        yConst = 70
    elif shape[1] > 1500:
        yConst = 50
    elif shape[1] > 1000:
        yConst = 30
    elif shape[1] > 500:
        yConst = 20
    else:
        yConst = 20
    
    if shape[0] > 4000:
        xConst = 250
    elif shape[0] > 3500:
        xConst = 250
    elif shape[0] > 3000:
        xConst = 200
    elif shape[0] > 2500:
        xConst = 100
    elif shape[0] > 2000:
        xConst = 70
    elif shape[0] > 1500:
        xConst = 50
    elif shape[0] > 1000:
        xConst = 30
    elif shape[0] > 500:
        xConst = 20
    else:
        xConst = 20
    

#     print(xConst, yConst)
#     print(shape)
    
# # #     print(rect)
    if rect[0,1] == rect[1,1]:
        rect[0,0] -= xConst
        rect[1,0] += xConst
    else:
        slope1 = (rect[1,1]-rect[0,1])/(rect[1,0]-rect[0,0])
        y1 = rect[0,1]
        x1 = rect[0,0]
        rect[0,0] -= xConst
        rect[1,0] += xConst
        rect[0,1] = y1 + (rect[0,0] - x1) * slope1
        rect[1,1] = y1 + (rect[1,0] - x1) * slope1
    
    if rect[2,1] == rect[3,1]:
        rect[2,0] += xConst
        rect[3,0] -= xConst
    else:
        slope3 = (rect[3,1]-rect[2,1])/(rect[3,0]-rect[2,0])
        y1 = rect[2,1]
        x1 = rect[2,0]
        rect[2,0] += xConst
        rect[3,0] -= xConst
        rect[2,1] = y1 + (rect[2,0] - x1) * slope3
        rect[3,1] = y1 + (rect[3,0] - x1) * slope3
    
    if rect[1,0] == rect[2,0]:
        rect[1,1] -= yConst
        rect[2,1] += yConst
    else:
        slope2 = (rect[2,1]-rect[1,1])/(rect[2,0]-rect[1,0])
        y1 = rect[1,1]
        x1 = rect[1,0]
        rect[1,1] -= yConst
        rect[2,1] += yConst
        rect[1,0] = x1 + (rect[1,1] - y1) / slope2
        rect[2,0] = x1 + (rect[2,1] - y1) / slope2
        
    if rect[0,0] == rect[3,0]:
        rect[0,1] -= yConst
        rect[3,1] += yConst
    else:
        slope4 = (rect[3,1]-rect[0,1])/(rect[3,0]-rect[0,0])
        y1 = rect[0,1]
        x1 = rect[0,0]
        rect[0,1] -= yConst
        rect[3,1] += yConst
        rect[0,0] = x1 + (rect[0,1] - y1) / slope4
        rect[3,0] = x1 + (rect[3,1] - y1) / slope4
        
#     print('after')
#     print(shape)
#     print(rect)
        
    # return the ordered coordinates
    return rect

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = pts.dtype)
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def getAreaOfInterest(originalImage):
    img = originalImage.copy()
    edges = cv2.Canny(img, 30, 150)
    
    padValue = max(originalImage.shape[0],originalImage.shape[1])//8
    
#     print(padValue)
    
    edges = np.pad(edges, padValue, constant_values=0)
    
#     openingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
#     opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, openingKernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(edges, kernel, iterations=1)
#     show_images([edges, dilate])
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    largestContour = np.zeros((0,1,2), dtype=contours[0].dtype)

    firstArea = 0
    for i in range(len(contours)):
        minAreaRect = cv2.minAreaRect(contours[i])
        points = cv2.boxPoints(minAreaRect)
        points = np.int0(points)
        p = order_points(points)
        height = math.sqrt((p[0, 0]-p[1,0])**2 + (p[0, 1]-p[1,1])**2)
        width = math.sqrt((p[0, 0]-p[3,0])**2 + (p[0, 1]-p[3,1])**2)

        area = width * height
                
        if i == 0:
            firstArea = area

        
        if  width > 20 and height > 20 and area > firstArea/5:
#             cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#             cp = cv2.drawContours(cp, [points],-1, (255,0,0), 3)
#             cp = cv2.drawContours(cp, contours[i],-1, (255,255,0), 5)
#             plt.imsave('test.png', cp)
#             show_images([cp], ['contour #' + str(i)])
            largestContour = np.vstack((largestContour,contours[i]))
#         print(ratio)
#         print(points)
        
#     print(contours[0])

    # find the perimeter of the first closed contour
#     perim = cv2.arcLength(largestContour, True)
#     # setting the precision
#     epsilon = 0.02*perim
#     # approximating the contour with a polygon
#     approxCorners = cv2.approxPolyDP(largestContour, epsilon, True)
#     print(approxCorners)
    
    temp = np.zeros((largestContour.shape[0],2), dtype=largestContour.dtype)
    for i in range(largestContour.shape[0]):
        temp[i] = largestContour[i,0]
        
#     minAreaRect = cv2.minAreaRect(approxCorners)
#     points = cv2.boxPoints(minAreaRect)
#     points = np.int0(points)
#     print(points)
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     cp = cv2.drawContours(cp, [points],-1, (255,0,0), 3)
#     show_images([cp])
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     for p in temp:
#         cv2.circle(cp, (int(p[0]),int(p[1])), 100, (0, 0, 255), -1)
    
#     show_images([cp])
    
        
#     print(temp.shape)
#     s = np.sum(temp, axis=1)
#     diff = np.diff(temp, axis=1)
#     print(np.sort(s))
#     print(np.sort(diff.T))
    
    
    firstRow = originalImage[0,:]
    lastRow = originalImage[originalImage.shape[0]-1,:]
    firstColumn = originalImage[:,0]
    lastColumn = originalImage[:,originalImage.shape[1]-1]
    
    t = np.mean(firstRow)
    firstRow[firstRow<t] = t
    t = np.mean(lastRow)
    lastRow[lastRow<t] = t
    t = np.mean(firstColumn)
    firstColumn[firstColumn<t] = t
    t = np.mean(lastColumn)
    lastColumn[lastColumn<t] = t
    
#     print(originalImage.shape)
#     print(pts)
#     originalImage = np.pad(originalImage, 100, mode="edge")
    originalImage = np.pad(originalImage, padValue, mode="edge")

    pts = getPts(temp, originalImage.shape)
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     cp = cv2.drawContours(cp, largestContour,-1, (255,0,0), 100)

#     for p in pts:
#         print(p)
#         cv2.circle(cp, (int(p[0]),int(p[1])), 100, (0, 0, 255), -1)
#         show_images([cp])
    warped = four_point_transform(originalImage, pts)
    
    return warped

def getAreaOfInterest2(originalImage):
    img = originalImage.copy()
    edges = cv2.Canny(img, 30, 150)
    
    padValue = max(originalImage.shape[0],originalImage.shape[1])//8
    
#     print(padValue)
    edges = np.pad(edges, padValue, constant_values=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edges, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    largestContour = np.zeros((0,1,2), dtype=contours[0].dtype)

    firstArea = 0
    for i in range(len(contours)):
        minAreaRect = cv2.minAreaRect(contours[i])
        points = cv2.boxPoints(minAreaRect)
        points = np.int0(points)
        p = order_points(points)
        height = math.sqrt((p[0, 0]-p[1,0])**2 + (p[0, 1]-p[1,1])**2)
        width = math.sqrt((p[0, 0]-p[3,0])**2 + (p[0, 1]-p[3,1])**2)

        area = width * height
                
        if i == 0:
            firstArea = area

        if  width > 20 and height > 20 and area > firstArea/5:
            largestContour = np.vstack((largestContour,contours[i]))
    
    
    firstRow = originalImage[0,:]
    lastRow = originalImage[originalImage.shape[0]-1,:]
    firstColumn = originalImage[:,0]
    lastColumn = originalImage[:,originalImage.shape[1]-1]
    
    t = np.mean(firstRow)
    firstRow[firstRow<t] = t
    t = np.mean(lastRow)
    lastRow[lastRow<t] = t
    t = np.mean(firstColumn)
    firstColumn[firstColumn<t] = t
    t = np.mean(lastColumn)
    lastColumn[lastColumn<t] = t    
    
    minAreaRect = cv2.minAreaRect(largestContour)
    points = cv2.boxPoints(minAreaRect)
    points = np.int0(points)
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     cp = cv2.drawContours(cp, [points],-1, (255,0,0), 10)
#     show_images([cp])
    originalImage = np.pad(originalImage, padValue, mode="edge")
    pts = getPts(points, originalImage.shape)
#     print(pts)
#     print(points)
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     cp = cv2.drawContours(cp, [pts.astype(points.dtype)],-1, (255,0,0), 10)
#     show_images([cp])
    warped = four_point_transform(originalImage, pts)
#     show_images([warped])
    return warped

def checkVertical(img):
    # height = img.shape[0]
    # width = img.shape[1]
    
    # widthHalf = 50
    # heightHalf = 50
#     print(width, height, widthHalf, heightHalf)
    edges = cv2.Canny(img,30,150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(edges, kernel, iterations=1)
#     show_images([dilate])
    
#     cdst = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     lines = cv2.HoughLines(dilate, 1, np.pi / 180, max(width, height)//2, None, 0, 0)
#     angles = np.round(np.degrees(lines[:,:,1])).astype(np.int16)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    hspace, angles, dists = hough_line(dilate, tested_angles)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists)

    angles = np.round(np.degrees(angles))
    (values,counts) = np.unique(angles,return_counts=True)
    theta = values[np.argmax(counts)]
    
    rotationAngle = 0
    
    if 0 <= theta <= 45:
        rotationAngle = 90 + theta
    elif 45 < theta < 90:
        rotationAngle = theta - 90
    elif -90 < theta < 0:
        rotationAngle = 90 + theta
        
#     print(img)
    # show_images([img])
    # print(theta, rotationAngle)
    rotated = rotate(img, rotationAngle, mode="edge", resize=True)
    # show_images([rotated])
    # print(angles)
#     print(theta)

#     if lines is not None:
#         for i in range(0, len(lines)):
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
# #             print(math.degrees(theta))
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
#     show_images([dilate, cdst])
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dilate = cv2.dilate(edges, kernel, iterations=1)
#     show_images([img,edges])
#     rows = edges[height//2 - heightHalf: height//2 + heightHalf, :]
#     columns = edges[:, width//2 - widthHalf: width//2 + widthHalf]
#     rowsSum = np.sum(rows, axis=1)
#     columnsSum = np.sum(columns, axis=0)
# #     print(rows)
# #     print(columns)
#     if np.max(columnsSum) > np.max(rowsSum):
#         img = np.rot90(img, axes=(0,1))

    return (rotated*255).astype(np.uint8)

def rotateImage(img):
    return checkVertical(getAreaOfInterest(getAreaOfInterest(getAreaOfInterest2(img))))

    