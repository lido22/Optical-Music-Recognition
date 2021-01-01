from commonfunctions import *

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
