from commonfunctions import *

def ignoreInterStaffs(filteredImg,lines, staffHeight, spaceHeight):
    r = np.linspace(0, filteredImg.shape[0], filteredImg.shape[0]-1, endpoint=False).astype(np.uint32)
    rng = lines[0]-spaceHeight//2
    rem = np.linspace(0, rng, rng+1).astype(np.uint32)
    r = np.setdiff1d(r, rem)
    
    for i in range(4,len(lines), 5):
        if i != len(lines)-1:
            start = lines[i]+spaceHeight//2
            end = lines[i+1]-spaceHeight//2
            rem = np.linspace(start, end, end+1-start, endpoint=True).astype(np.uint32)
            r = np.setdiff1d(r, rem)
        else:
            start = lines[i]+spaceHeight//2
            end = filteredImg.shape[0]-1
            rem = np.linspace(start, end, end+1-start, endpoint=True).astype(np.uint32)
            r = np.setdiff1d(r, rem)
    for i in range(len(lines)):
        start = lines[i] - staffHeight + 1
        end = lines[i] + staffHeight - 1
        rem = np.linspace(start, end, end+1-start, endpoint=True).astype(np.uint32)
        r = np.setdiff1d(r, rem)
            
    filteredImg[r,:] = 1

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
        end = min(rows, begin+length+spaceHeight+staffHeight)
        start2 = min(rows, begin+length+spaceHeight)
        if np.any(filteredImg[start:start+staffHeight, col] == 0) or np.any(filteredImg[start2:end, col] == 0):
            verConnected = True

        if not (verConnected):
            img[i[1]:i[1]+i[2], i[0]] = 1
            v.remove(i)
            eliminated.append(i)
    for i in v:
        staffs[i[1]:i[1]+i[2],i[0]] = 0
    return staffs, v, eliminated