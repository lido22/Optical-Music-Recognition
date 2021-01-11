from commonfunctions import *

def getTheta(point1, point2):
    with np.errstate(divide='ignore'):
        m = (point2[1]-point1[1])/(point2[0]-point1[0])
    return math.atan(m)


def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)
    
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
    
    # cp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for p in rect:
    #     cv2.circle(cp, (int(p[0]),int(p[1])), 20, (0, 0, 255), -1)

    # show_images([cp])

    theta1 = getTheta(rect[0], rect[1])
    theta2 = getTheta(rect[1], rect[2])
    theta3 = getTheta(rect[3], rect[2])
    theta4 = getTheta(rect[0], rect[3])

    topPoint = pts[np.argmin(pts[:,1])]
    bottomPoint = pts[np.argmax(pts[:,1])]

    leftPoint = pts[np.argmin(pts[:,0])]
    rightPoint = pts[np.argmax(pts[:,0])]

    # print(shape)
    topDistance = lineseg_dists(topPoint, np.array([rect[0]]), np.array([rect[1]]))[0]
    topDistance = math.ceil(topDistance) + 20

    bottomDistance = lineseg_dists(bottomPoint, np.array([rect[3]]), np.array([rect[2]]))[0]
    bottomDistance = math.ceil(bottomDistance) + 20
    
    leftDistance = lineseg_dists(leftPoint, np.array([rect[0]]), np.array([rect[3]]))[0]
    leftDistance = math.ceil(leftDistance) + 10
    
    rightDistance = lineseg_dists(rightPoint, np.array([rect[1]]), np.array([rect[2]]))[0]
    rightDistance = math.ceil(rightDistance) + 10
    
    # print(rect[0], rect[1], rect[2], rect[3])    
    # print(math.degrees(theta1),math.degrees(theta2),math.degrees(theta3),math.degrees(theta4))
    # print(topPoint, bottomPoint, leftPoint, rightPoint)
    # print(topDistance, bottomDistance, leftDistance, rightDistance)

    rect[0,0] -= abs(math.cos(theta1) * leftDistance) - abs(math.cos(theta4) * topDistance)
    rect[0,1] += abs(math.sin(theta1) * leftDistance) - abs(math.sin(theta4) * topDistance)

    # cv2.circle(cp, (int(rect[0,0]),int(rect[0,1])), 20, (255, 0, 0), -1)
    # show_images([cp])

    rect[1,0] += abs(math.cos(theta1) * rightDistance) - abs(math.cos(theta2) * topDistance)
    rect[1,1] += abs(math.sin(theta1) * rightDistance) - abs(math.sin(theta2) * topDistance)

    # cv2.circle(cp, (int(rect[1,0]),int(rect[1,1])), 20, (255, 0, 0), -1)
    # show_images([cp])

    rect[2,0] += abs(math.cos(theta2) * bottomDistance) + abs(math.cos(theta3) * rightDistance)
    rect[2,1] += abs(math.sin(theta2) * bottomDistance) + abs(math.sin(theta3) * rightDistance)

    # cv2.circle(cp, (int(rect[2,0]),int(rect[2,1])), 20, (255, 0, 0), -1)
    # show_images([cp])

    rect[3,0] -= abs(math.cos(theta3) * leftDistance) + abs(math.cos(theta4) * bottomDistance)
    rect[3,1] -= abs(math.sin(theta3) * leftDistance) - abs(math.sin(theta4) * bottomDistance)

    # cv2.circle(cp, (int(rect[3,0]),int(rect[3,1])), 20, (255, 0, 0), -1)
    # show_images([cp])

    
#     width = np.linalg.norm(rect[0]-rect[1])
#     height = np.linalg.norm(rect[1]-rect[2])
#     print(rect)
# #     print(shape)
#     const = min(shape[1],shape[0])/15
#     xConst = 0
#     yConst = 0
    
#     if shape[1] > 4000:
#         yConst = 250
#     elif shape[1] > 3500:
#         yConst = 250
#     elif shape[1] > 3000:
#         yConst = 200
#     elif shape[1] > 2500:
#         yConst = 100
#     elif shape[1] > 2000:
#         yConst = 70
#     elif shape[1] > 1500:
#         yConst = 50
#     elif shape[1] > 1000:
#         yConst = 30
#     elif shape[1] > 500:
#         yConst = 20
#     else:
#         yConst = 20
    
#     if shape[0] > 4000:
#         xConst = 250
#     elif shape[0] > 3500:
#         xConst = 250
#     elif shape[0] > 3000:
#         xConst = 200
#     elif shape[0] > 2500:
#         xConst = 100
#     elif shape[0] > 2000:
#         xConst = 70
#     elif shape[0] > 1500:
#         xConst = 50
#     elif shape[0] > 1000:
#         xConst = 30
#     elif shape[0] > 500:
#         xConst = 20
#     else:
#         xConst = 20
    

# #     print(xConst, yConst)
# #     print(shape)
    
# # # #     print(rect)
#     if rect[0,1] == rect[1,1]:
#         rect[0,0] -= xConst
#         rect[1,0] += xConst
#     else:
#         slope1 = (rect[1,1]-rect[0,1])/(rect[1,0]-rect[0,0])
#         y1 = rect[0,1]
#         x1 = rect[0,0]
#         rect[0,0] -= xConst
#         rect[1,0] += xConst
#         rect[0,1] = y1 + (rect[0,0] - x1) * slope1
#         rect[1,1] = y1 + (rect[1,0] - x1) * slope1
    
#     if rect[2,1] == rect[3,1]:
#         rect[2,0] += xConst
#         rect[3,0] -= xConst
#     else:
#         slope3 = (rect[3,1]-rect[2,1])/(rect[3,0]-rect[2,0])
#         y1 = rect[2,1]
#         x1 = rect[2,0]
#         rect[2,0] += xConst
#         rect[3,0] -= xConst
#         rect[2,1] = y1 + (rect[2,0] - x1) * slope3
#         rect[3,1] = y1 + (rect[3,0] - x1) * slope3
    
#     if rect[1,0] == rect[2,0]:
#         rect[1,1] -= yConst
#         rect[2,1] += yConst
#     else:
#         slope2 = (rect[2,1]-rect[1,1])/(rect[2,0]-rect[1,0])
#         y1 = rect[1,1]
#         x1 = rect[1,0]
#         rect[1,1] -= yConst
#         rect[2,1] += yConst
#         rect[1,0] = x1 + (rect[1,1] - y1) / slope2
#         rect[2,0] = x1 + (rect[2,1] - y1) / slope2
        
#     if rect[0,0] == rect[3,0]:
#         rect[0,1] -= yConst
#         rect[3,1] += yConst
#     else:
#         slope4 = (rect[3,1]-rect[0,1])/(rect[3,0]-rect[0,0])
#         y1 = rect[0,1]
#         x1 = rect[0,0]
#         rect[0,1] -= yConst
#         rect[3,1] += yConst
#         rect[0,0] = x1 + (rect[0,1] - y1) / slope4
#         rect[3,0] = x1 + (rect[3,1] - y1) / slope4
        
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
    # show_images([edges])
    edges = np.pad(edges, padValue, constant_values=0)
    # show_images([edges])
#     openingKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
#     opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, openingKernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(edges, kernel, iterations=1)
    # show_images([edges, dilate])
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
            # cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
            # cp = cv2.drawContours(cp, [points],-1, (255,0,0), 10)
            # cp = cv2.drawContours(cp, contours[i],-1, (255,255,0), 5)
            # # plt.imsave('test.png', cp)
            # show_images([cp], ['contour #' + str(i)])
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
    # cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
    # cp = cv2.drawContours(cp, largestContour,-1, (255,0,0), 10)
    # show_images([originalImage, cp])
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     for p in temp:
#         cv2.circle(cp, (int(p[0]),int(p[1])), 100, (0, 0, 255), -1)
    
#     show_images([cp])
    
        
#     print(temp.shape)
#     s = np.sum(temp, axis=1)
#     diff = np.diff(temp, axis=1)
#     print(np.sort(s))
#     print(np.sort(diff.T))
    
    
    
#     print(originalImage.shape)
#     print(pts)
    
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

    originalImage = np.pad(originalImage, padValue, mode="edge")
    # originalImage = np.pad(originalImage, padValue, mode="edge")

    pts = getPts(temp, originalImage.shape)
    
    # xmin = np.min(pts[:,0])
    # xmax = np.max(pts[:,0])
    # ymin = np.min(pts[:,1])
    # ymax = np.max(pts[:,1])
    
    # if xmax > originalImage.shape[1]:
    #     lastColumn = originalImage[:,originalImage.shape[1]-1]
    #     t = np.mean(lastColumn)
    #     lastColumn[lastColumn<t] = np.max(lastColumn)    
    #     originalImage = np.pad(originalImage, ((0,0), (0, int(xmax-originalImage.shape[1]))), mode="edge")

    # if ymax > originalImage.shape[0]:
    #     lastRow = originalImage[originalImage.shape[0]-1,:]
    #     t = np.mean(lastRow)
    #     lastRow[lastRow<t] = np.max(lastRow)    
    #     originalImage = np.pad(originalImage, ((0, int(ymax-originalImage.shape[0])), (0, 0)), mode="edge")

    # if xmin < 0:
    #     firstColumn = originalImage[:,0]
    #     t = np.mean(firstColumn)
    #     firstColumn[firstColumn<t] = np.max(firstColumn)    
    #     originalImage = np.pad(originalImage, (((0,0), (int(np.abs(xmin)), 0))), mode="edge")
    #     pts[:,0] += abs(xmin)

    # if ymin < 0:
    #     firstRow = originalImage[0,:]
    #     t = np.mean(firstRow)
    #     firstRow[firstRow<t] = np.max(firstRow)    
    #     originalImage = np.pad(originalImage, ((int(abs(ymin)), 0), (0, 0)), mode="edge")
    #     pts[:,1] += abs(ymin)

#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     cp = cv2.drawContours(cp, largestContour,-1, (255,0,0), 100)

#     for p in pts:
#         print(p)
#         cv2.circle(cp, (int(p[0]),int(p[1])), 100, (0, 0, 255), -1)
#         show_images([cp])
    warped = four_point_transform(originalImage, pts)
    # show_images([warped])    
    return warped

def getAreaOfInterest2(originalImage):
    img = originalImage.copy()
    edges = cv2.Canny(img, 30, 150)
    # show_images([edges])
#     padValue = max(originalImage.shape[0],originalImage.shape[1])//8
    
# #     print(padValue)
#     edges = np.pad(edges, padValue, constant_values=0)
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
    
    
    # firstRow = originalImage[0,:]
    # lastRow = originalImage[originalImage.shape[0]-1,:]
    # firstColumn = originalImage[:,0]
    # lastColumn = originalImage[:,originalImage.shape[1]-1]
    
    # t = np.mean(firstRow)
    # firstRow[firstRow<t] = t
    # t = np.mean(lastRow)
    # lastRow[lastRow<t] = t
    # t = np.mean(firstColumn)
    # firstColumn[firstColumn<t] = t
    # t = np.mean(lastColumn)
    # lastColumn[lastColumn<t] = t    
    
    rect = cv2.minAreaRect(largestContour)
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    # cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
    # cp = cv2.drawContours(cp, [points],-1, (255,0,0), 1)
    # # show_images([cp])
    # plt.imsave('contour.png', cp)
    
    # print(points)
    points = order_points(points).astype(np.float32)
    
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    xmin = np.min(points[:,0])
    xmax = np.max(points[:,0])
    ymin = np.min(points[:,1])
    ymax = np.max(points[:,1])
    
    # print(originalImage.shape)
    # print(xmin,xmax,ymin,ymax)
    
    if xmax > originalImage.shape[1]:
        lastColumn = originalImage[:,originalImage.shape[1]-1]
        t = np.mean(lastColumn)
        lastColumn[lastColumn<t] = t    
        originalImage = np.pad(originalImage, ((0,0), (0, int(xmax-originalImage.shape[1]))), mode="edge")

    if ymax > originalImage.shape[0]:
        lastRow = originalImage[originalImage.shape[0]-1,:]
        t = np.mean(lastRow)
        lastRow[lastRow<t] = t    
        originalImage = np.pad(originalImage, ((0, int(ymax-originalImage.shape[0])), (0, 0)), mode="edge")

    if xmin < 0:
        firstColumn = originalImage[:,0]
        t = np.mean(firstColumn)
        firstColumn[firstColumn<t] = t    
        originalImage = np.pad(originalImage, (((0,0), (int(np.abs(xmin)), 0))), mode="edge")
        points[:,0] += abs(xmin)

    if ymin < 0:
        firstRow = originalImage[0,:]
        t = np.mean(firstRow)
        firstRow[firstRow<t] = t    
        originalImage = np.pad(originalImage, ((int(abs(ymin)), 0), (0, 0)), mode="edge")
        points[:,1] += abs(ymin)

    warped = four_point_transform(originalImage, points)
    
    # plt.imsave('contour2.png', warped)
    
    # print(points)
    # plt.imsave('img3.png', originalImage)
    # originalImage = np.pad(originalImage, padValue, mode="edge")
    # plt.imsave('img4.png', originalImage)
    # pts = getPts(points, originalImage.shape)
#     print(pts)
#     print(points)
#     cp = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
#     cp = cv2.drawContours(cp, [pts.astype(points.dtype)],-1, (255,0,0), 10)
#     show_images([cp])
    # plt.imsave('img5.png', originalImage)
    # warped = four_point_transform(originalImage, pts)
    # plt.imsave('img6.png', warped)

#     show_images([warped])
    # show_images([warped])    
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
    # print(rotationAngle)
    # print(theta, rotationAngle)

    firstRow = img[0,:]
    lastRow = img[img.shape[0]-1,:]
    firstColumn = img[:,0]
    lastColumn = img[:,img.shape[1]-1]
    
    t = np.mean(firstRow)
    firstRow[firstRow<t] = t
    t = np.mean(lastRow)
    lastRow[lastRow<t] = t
    t = np.mean(firstColumn)
    firstColumn[firstColumn<t] = t
    t = np.mean(lastColumn)
    lastColumn[lastColumn<t] = t    

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
    # show_images([rotated])    
    return (rotated*255).astype(np.uint8)

def rotateImage(img):
    return checkVertical(getAreaOfInterest(getAreaOfInterest2(img)))
