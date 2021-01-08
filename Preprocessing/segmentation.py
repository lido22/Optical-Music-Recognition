from commonfunctions import *

def getLines(img, staffHeight, spaceHeight):
    rows_sum = np.sum(img, axis=1)
    lines, _ = find_peaks(rows_sum, height = 0.5*img.shape[1], distance=spaceHeight+staffHeight//2)
    return lines

def segmentImage(lines, spaceHeight, height):
    detected_lines = np.zeros((height,))
    detected_lines[lines] = 1

    starts = np.array((detected_lines[:-1] == 1) & (detected_lines[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends = np.array((detected_lines[:-1] == 0) & (detected_lines[1:] == 1))
    ends_ix = np.where(ends)[0]

    starts_ix = starts_ix[:-1]
    ends_ix = ends_ix[1:]

    halfs = [0]

    for i in range(len(starts_ix)):
        diff = ends_ix[i] - starts_ix[i]
        if diff > 5 * spaceHeight:
            halfs.append((ends_ix[i] + starts_ix[i])//2)

    halfs.append(height-1)
    
    return halfs


def getObjects(staffless, lines, halfY, staffHeight, spaceHeight, j):
#     print(staffHeight, spaceHeight)
    cnt, hir = cv2.findContours(staffless, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = []
    
    for i in range(len(cnt)):
#         print(i)
        if hir[0,i,3] == -1:
            contours.append(cnt[i])
    
            
    boundingRects = [cv2.boundingRect(c) for c in contours]
# #     boundingRects = sorted(boundingRects, key=lambda b: b[1])  
    boundingRects = sorted(boundingRects, key=cmp_to_key(mycmp))
        
    (x,y,w,h) = boundingRects[0]
    mergedRects = [(x,y,x+w,y+h)]
    k = 0
    for i in range(1, len(boundingRects)):
        (x,y,w,h) = boundingRects[i]
        old_x,old_y,old_x2,old_y2 = mergedRects[k]
        old_w = old_x2 - old_x
        old_h = old_y2 - old_y
        
        print(i)
        print(old_x, old_y, old_x2, old_y2)
        print(old_w, old_h)
        print(x,y,x+w,y+h)
        print(w,h) 
        print()
        cp = cv2.cvtColor(staffless, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(cp, (old_x,old_y), (old_x2,old_y2), (0, 255, 0), 1) 
        cv2.rectangle(cp, (x,y), (x+w,y+h), (0, 0, 255), 1) 
        
        plt.imsave('it' + str(i) + '.png', cp)
        
        if (((abs(x - old_x2) < 0.75*spaceHeight or abs(old_x - x-w) < 0.75*spaceHeight) 
            and ((old_h < 3*staffHeight+spaceHeight and old_h - old_w > staffHeight)
             or (h < 3*staffHeight + spaceHeight and h - w > staffHeight)))
            or ((x >= old_x and x <= old_x2 and y >= old_y and y <= old_y2)
                or (x >= old_x and x <= old_x2 and y+h >= old_y and y+h <= old_y2)
                or (x+w >= old_x and x+w <= old_x2 and y >= old_y and y <= old_y2)
                or (x+w >= old_x and x+w <= old_x2 and y+h >= old_y and y+h <= old_y2))
           or (abs(x - old_x2) < staffHeight or abs(old_x - x-w) < staffHeight)):
            
            if h < spaceHeight//2 and w < spaceHeight//2:
                _, top, distanceTop = getNearestLine(y+halfY,lines)
#                 _, bottom, distanceBottom = getNearestLine(y+h+halfY,lines)
                if  (top == 2 and distanceTop < 0) or (top == 3 and distanceTop > 0):
                    mergedRects.append((x,y,x+w,y+h))
                    k += 1
                    continue

            mergedRects[k] = (min(old_x,x), min(old_y,y), max(old_x2,x+w), max(old_y2,y+h))
        else:
            mergedRects.append((x,y,x+w,y+h))
            k += 1
    
    cp = cv2.cvtColor(staffless, cv2.COLOR_GRAY2BGR)
    
    for b in mergedRects:
        cv2.rectangle(cp, (b[0],b[1]), (b[2],b[3]), (0, 255, 0), 1) 
    
#     for c in contours:
#         (x,y,w,h) = cv2.boundingRect(c)
#         min_x, max_x = x, x+w
#         min_y, max_y = y, y+h
#         cv2.rectangle(cp, (min_x,min_y), (max_x,max_y), (255, 0, 0), 1)

    plt.imsave('after' + str(j) + '.png', cp)

def mycmp(b1, b2):
    if (b2[0] >= b1[0] and b2[0] <= b1[2]+b1[0]) or (b1[0] >= b2[0] and b1[0] <= b2[2]+b2[0]):
        if b1[1] < b2[1]:
            return -1
        elif b1[1] > b2[1]:
            return 1
        return 0
    else:
        if b1[0] < b2[0]:
            return -1
        elif b1[0] > b2[0]:
            return 1
        return 0

def cmpWrap(cmpWrap):
    class K: 
        def __init__(self,obj):
            self.obj = obj
        def __lt__(self,other):
            b2 = self.obj
            b1 = other.obj
#             print(b2[0] ,b1[0] , b2[0], b1[2])
            if (b2[0] >= b1[0] and b2[0] <= b1[2]+b1[0]) or (b1[0] >= b2[0] and b1[0] <= b2[2]+b2[0]): # b2 intersect b1
#                 print(b1[1] ,b2[1])
                if b1[1] > b2[1]: # b1 is up and b2 is down
#                     print('true')
                    return True 
                else:
                    return False
            elif b2[0] < b1[0]:
                return True
            else:
                return False
    return K
    