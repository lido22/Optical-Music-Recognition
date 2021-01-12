from commonfunctions import *

from scipy.signal import convolve

import segmentation
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm
import cv2
import matplotlib.pyplot as plt
import os 
from skimage.measure import find_contours
from skimage.color import rgb2gray , rgba2rgb
from skimage.transform import resize
from PIL import Image
import sys
import random
from sklearn.model_selection import train_test_split
# import sklearn.datasets.images as loader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn import svm
import numpy as np
import argparse
from notes import getNoteCharacter
# import imutils  # If you are unable to install this library, ask the TA; we only need this in extract_hsv_histogram.
import cv2
import os
import random
import pickle

# load the classfier
file = open("nn2.pickle",'rb')
nn = pickle.load(file)

# classifier options
target_img_size = (32, 32) # fix image size because|> classification algorithms THAT WE WILL USE HERE expect that

# We are going to fix the random seed to make our experiments reproducible 
# since some algorithms use pseudorandom generators
random_seed = 42  
random.seed(random_seed)
np.random.seed(random_seed)


classes = ['a_1', 'a_16', 'a_2', 'a_32', 'a_4', 'a_8', 
           'barline ', 'chord', 'clef', '.', '&&', '##', '&', '', '#', '\meter<"4/2"> ', '\meter<"4/4"> ']

dataset_dir = sys.argv[1]
images = load_images_from_folder(dataset_dir)

fileNames = os.listdir(dataset_dir)

for i,img in enumerate(images):
    print(i)
    start_time = time.time()
    start = time.time()
    rotated = rotateImage(img)
    print('rotation time:' + str(time.time() - start))

    start = time.time()
    binary = binraization(rotated)//255
    print('binarization time:' + str(time.time() - start))
    
    # show_images([binary])

    start = time.time()
    staffHeight, spaceHeight = getRefLengths(binary)
    print('Ref lengths time:' + str(time.time() - start))
    
    start = time.time()
    filteredImg, candidates = getCandidateStaffs(binary, staffHeight)
    filteredImg, candidates, eliminated = removeLonelyStaffs(candidates, binary, staffHeight, spaceHeight, eliminated=[])
#     filteredImg1, candidates, eliminated = RemoveThinStaffs(candidates, filteredImg, staffHeight)
#     filteredImg2, candidates, eliminated = removeLonelyStaffs(candidates, binary, staffHeight, spaceHeight, eliminated)
#     filteredImg3, candidates = addFalseNegatives(candidates, filteredImg2, staffHeight, staffHeight, eliminated)
#     # print(filteredImg3)
    staffLess = (binary-filteredImg).astype(np.uint8)
    print('staff removal time:' + str(time.time() - start))

    start = time.time()
    lines = getLines(1-filteredImg, staffHeight, spaceHeight)
    staff = cv2.cvtColor(staffLess, cv2.COLOR_GRAY2BGR)
    staff[lines] = [0,0,255]
    plt.imsave('staff'  +'.png', staff)
    print('getting lines time:' + str(time.time() - start))
    
    objects = segmentImage(staffLess, lines, staffHeight, spaceHeight)
    firstTime = True
    output = ""
    perviousAccedental = ""
    for o in objects:
        features = extract_hog_features(staffLess[o[1]:o[3], o[0]:o[2]],target_img_size)
        symbol_name = classes[np.argmax( nn.predict_proba([features]))]

#         show_images([staffLess[o[1]:o[3], o[0]:o[2]]])
        if symbol_name == "a_2" or symbol_name == "a_4":
            if isHalf(staffLess[o[1]:o[3], o[0]:o[2]],spaceHeight) :
                symbol_name = "a_2"
            else:
                symbol_name = "a_4"
        if symbol_name =="clef":
            if firstTime:
                firstTime = False
                output+= '['
            else:
                output+= ']\n['
        elif firstTime:
            continue 
        #beam
        elif (o[2]-o[0]) > 4*spaceHeight:
            try:
                output += getNoteCharacter(staffLess, o, "beam", lines, staffHeight, spaceHeight)+" "
            except:
                continue
        #dot and barline
        elif symbol_name == "." or symbol_name == "barline ":
                if isDot(staffLess[o[1]:o[3], o[0]:o[2]],spaceHeight):
                    output += "."
        
        #chord
        elif symbol_name == "chord":
#             print("chord")
            try:
                notes = getchordText([o[1],o[3]],staffLess[o[1]:o[3], o[0]:o[2]],staffHeight,spaceHeight,lines)
            except:
                noteSymbol = getNoteCharacter(staffLess, o, "a_4", lines, staffHeight, spaceHeight)
       
                # print(noteSymbol)
                output += noteSymbol[0]+perviousAccedental+noteSymbol[1:]+" "
                perviousAccedental = ""
                # print('contin')

                continue
            output +="{"
            for k in range(0,len(notes)-2,2):
                output += notes[k:k+2]+"/4,"
            output += notes[-2:]+"/4"
            output+= "} "
            
        #note
        elif symbol_name!="" and  symbol_name[0]=="a" :
            try:
                noteSymbol = getNoteCharacter(staffLess, o, symbol_name, lines, staffHeight, spaceHeight)
                output += noteSymbol[0]+perviousAccedental+noteSymbol[1:]+" "
                perviousAccedental = ""
            except:
                # print('contin')
                continue
        #accedentals
        elif symbol_name == '\meter<"4/2"> ' or symbol_name == '\meter<"4/4"> ':
            output += symbol_name
        else:

            perviousAccedental= symbol_name
    output+="]"
    print(i)
    if len(output.split("\n"))>1:
        output ="{\n"+output+"\n}"
    f =open(sys.argv[2]+"/"+fileNames[i].split(".")[0]+".txt",'w') 
    f.write(output)
    f.close()
    print("time: %s seconds" % (time.time() - start_time))
    print("---------------------------------------------------------")


