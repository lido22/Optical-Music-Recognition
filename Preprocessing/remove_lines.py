from commonfunctions import *

def ignoreInterStaffs(filteredImg,lines, spaceHeight):
    
    cols = np.sum(filteredImg, axis=1)
    r = np.where(cols>filteredImg.shape[1]//1.3)[0]
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
            
    filteredImg[r,:] = 1

def getCandidateStaffs(binaryImg, staffHeight):
    filteredImg = np.copy(binaryImg)
    cols = filteredImg.shape[1]
    rows = filteredImg.shape[0]
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
                
    return filteredImg

