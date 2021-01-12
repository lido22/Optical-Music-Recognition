from commonfunctions import * 
def extract_hog_features(img,target_img_size):
    """
    TODO
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()
def extract_features(img,target_img_size, feature_set='hog'):
    """
    TODO
    Given either 'hsv_hist', 'hog', 'raw', call the respective function and return its output
    """
    if(feature_set =="hog"):
        return extract_hog_features(img)
    if(feature_set =="hsv_hist"):
        return extract_hsv_histogram(img)
    if(feature_set =="raw"):
        return extract_raw_pixels(img)

def isDot(img, spaceHeight):
    if img.shape[0] > spaceHeight:
        return False
    return True
def chord2text(img,cnt_pos,staffHeight,spaceHeight,lines):
    char_middle = ''
    char_top = ''
    char_down = ''
    height = cnt_pos[1] - cnt_pos[0]
    # show_images([img])
    # print(img.shape)
    if height > 3 * spaceHeight: 
        # height of 3 notes
        # print('cord 3 notes')
        # print(cnt_pos,spaceHeight)
        # show_images([img[:cnt_pos[1]+spaceHeight,:]])
        # space = (img.shape[0] - 3 * spaceHeight -2*staffHeight )//2
        space = spaceHeight
        middle = img[spaceHeight+staffHeight*3//2: 2*(spaceHeight+staffHeight*3//2),:]
        # show_images([img])

        # show_images([img[:spaceHeight+staffHeight,:],middle,img[2*(spaceHeight+staffHeight*3//2)-staffHeight//2:,:]],['top','middle','down'])
        #get top and down
        _, top, distanceTop = getNearestLine(cnt_pos[0],lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[0] + spaceHeight+staffHeight*3//2 ,lines)
        char_top = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)

        _, top, distanceTop = getNearestLine(cnt_pos[0] + 2*(spaceHeight+staffHeight*3//2)-staffHeight//2,lines)
        _, bottom, distanceBottom = getNearestLine((cnt_pos[1]),lines)
        char_down = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        if np.sum(middle) > (middle.shape[0]*middle.shape[1])//2:
            # there's a note in middle 
            # print('middle exists')
            _, top, distanceTop = getNearestLine(cnt_pos[0] + spaceHeight+staffHeight*3//2 ,lines)
            _, bottom, distanceBottom = getNearestLine(cnt_pos[0] + 2*(spaceHeight+staffHeight*3//2),lines)
            char_middle = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
    elif height < 1.5 * spaceHeight:
        # print('cord one note')
        # heigth of 1 note
        # don't think it shoud come here but anyway
        _, top, distanceTop = getNearestLine(cnt_pos[0],lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[1],lines)
        char_down = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
    else: 
        # print('cord two notes')
        # show_images([img[:spaceHeight+staffHeight*3//2,:],img[img.shape[0]-spaceHeight-staffHeight*3//2:,:]],['top','down'])
        _, top, distanceTop = getNearestLine(cnt_pos[0],lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[0]+ spaceHeight+staffHeight*3//2,lines)
        char_top = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        _, top, distanceTop = getNearestLine(cnt_pos[1]-spaceHeight-staffHeight*3//2,lines)
        _, bottom, distanceBottom = getNearestLine(cnt_pos[1],lines)
        char_down = getHeadCharacter(top, distanceTop, bottom, distanceBottom, spaceHeight)
        # height of 2 notes
    # print(str(char_down) + str(char_middle) + str(char_top))
    return [str(char_down) , str(char_middle) , str(char_top)]
def extract_cnt(cnt_img,staffHeight,spaceHeight):
    chord = cnt_img.copy()
    chord[chord>0] = 1
    # print(chord.shape)
    chord = chord[:,chord.shape[1]//2:]
    idx = np.where(np.sum(chord[:,:staffHeight],axis= 1)>staffHeight//2)
    # print(idx)
    min_y= idx[0][0]
    max_y= idx[0][len(idx[0])-1]
    # print(min_y,max_y)
    # cnt_img = cnt_img[min_y:max_y,:]
    # cnt_img[cnt_img.shape[0]//2,:] = 0
    # show_images([cnt_img])
    return min_y, max_y
def getchordText(cnt_pos,cnt_img,staffHeight,spaceHeight,lines):
    cnt_img[cnt_img > 0]=1 
    temp = binary_dilation(cnt_img[:spaceHeight,:].copy(),np.ones((1,staffHeight)))
    h_hist = np.sum(temp,axis=0)
    bar_idx = np.where(h_hist== np.max(h_hist))[0][0]
    img = cnt_img.copy()
    # print(cnt_img.shape,bar_idx,cnt_img.shape[1]//4)
    if ((bar_idx >  cnt_img.shape[1]//3 ) and (bar_idx < cnt_img.shape[1]*2//3)):
        # print('cord two sides')
        # chord is two sides 
        # show_images([img])
        rh = img[:,:bar_idx] #right half
        # rh = binary_opening(rh.copy(),np.ones((1,rh.shape[1]//2)))
        lh = img[:,bar_idx:] # left half 
        # lh = binary_opening(lh.copy(),np.ones((1,lh.shape[1]//2)))
        # show_images([rh,lh])
        # apply on right half
        min_y ,max_y = extract_cnt(rh,staffHeight,spaceHeight)
        # show_images([rh[min_y:max_y,:]],['two sides Right side'])
        rcnt_pos = [min_y+cnt_pos[0],max_y+cnt_pos[0]]
        text1 = chord2text(rh[min_y:max_y,:],rcnt_pos,staffHeight,spaceHeight,lines)
        #apply on left half
        min_y ,max_y = extract_cnt(lh,staffHeight,spaceHeight)
        # show_images([lh[min_y:max_y,:]],['two sides left side'])
        lcnt_pos = [min_y+cnt_pos[0],max_y+cnt_pos[0]]     
        text2 = chord2text(lh[min_y:max_y,:],lcnt_pos,staffHeight,spaceHeight,lines)
        text1.extend(text2)
        op = []
        for t in text1: 
            if t!= '':
                op.append(t)
        op = sorted(op,key= lambda b:b[0])
        # print(op)
        return "".join(op)
    else:
        # print('cord one side')
        # chor is one side
        min_y ,max_y = extract_cnt(cnt_img,staffHeight,spaceHeight)
        # show_images([cnt_img[min_y:max_y,:]],['one side'])
        cnt_pos = [min_y+cnt_pos[0],max_y+cnt_pos[0]]
        text1 = chord2text(cnt_img[min_y:max_y,:],cnt_pos,staffHeight,spaceHeight,lines)
        op = []
        for t in text1: 
            if t != '':
                op.append(t)
        op = sorted(op,key= lambda b:b[0])
        return "".join(op)
def isHalf(img, spaceHeight):

    w = img.shape[1]
    h = img.shape[0]
    hist = np.zeros((w,4), dtype=np.uint32)
    min_x, max_x = 0, w
    min_y, max_y = 0, h 
    for i in range(w):
        window = img[:, i: min(i + 1, w)]
    #     show_images([window])
        xprojection = np.sum(window, axis=1)
    #     xprojection = np.where(xprojection>spaceHeight//4, 1,0)

        starts = np.array((xprojection[:-1] == 0) & (xprojection[1:] != 0))
        starts_ix = np.where(starts)[0] + 1
        ends = np.array((xprojection[:-1] != 0) & (xprojection[1:] == 0))
        ends_ix = np.where(ends)[0]

        if xprojection[0] != 0:
            starts_ix = np.append(0, starts_ix)

        if xprojection[-1] != 0:
            ends_ix = np.append(ends_ix, xprojection.size-1)

        if starts_ix.size != 0:
            index = np.argmax(ends_ix - starts_ix)
            hist[i,1] = min_x + i
            hist[i,2] = min_y + starts_ix[index]
            hist[i,3] = min_y + ends_ix[index]
            length = hist[i,3] - hist[i,2]
          
            if 0.5*spaceHeight < length < spaceHeight*1.5:
                hist[i,0] = length
    projections = len(np.where(hist[:,0]>0)[0])
    if projections > img.shape[1]//3:
        return False
    else:
        return True

def downSize(image, width=1000):
    (h, w) = image.shape[:2]
    # print(h, w)
    shrinkingRatio = width / float(w)
    dsize  = (width, int(h * shrinkingRatio))
    resized = cv2.resize(image, dsize , interpolation=cv2.INTER_AREA)
    return resized  
