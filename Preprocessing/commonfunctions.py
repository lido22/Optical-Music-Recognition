import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
from collections import Counter
import skimage.filters as fr
import skimage as sk
import cv2, time, os, math
from skimage.transform import hough_line, hough_line_peaks, rotate
from rotation import rotateImage
from remove_lines import getCandidateStaffs, removeLonelyStaffs
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv, rgba2rgb
# Convolution:
from scipy.signal import convolve2d, find_peaks, peak_widths
from scipy import fftpack
from skimage.util import random_noise
from skimage.filters import median, threshold_otsu
from skimage.feature import canny
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from notes import getNoteCharacter, getHeadCharacter
from segmentation import getObjects, getLines, getHalfs, segmentImage
# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt
from functools import cmp_to_key

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles..
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

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
        starts = np.array((a[:-1] != 0) & (a[1:] == 0))
        starts_ix = np.where(starts)[0] + 2
        ends = np.array((a[:-1] == 0) & (a[1:] != 0))
        ends_ix = np.where(ends)[0] + 2
        s1 = starts_ix.size
        s2 = ends_ix.size
        
        if a[0] == 0:
            starts_ix = np.append(1, starts_ix)

        if a[-1] == 0:
            ends_ix = np.append(ends_ix, a.size+1)
        
#         if s2 > s1:
#             starts_ix = np.pad(starts_ix,(s2-s1,0), mode='constant', constant_values=(1))
#         elif s1 > s2:
#             ends_ix = np.pad(ends_ix,(0,s1-s2), mode='constant', constant_values=(a.size + 1))
#         elif s1 > 0 and s2 > 0 and starts_ix[0] > ends_ix[0]:
#             starts_ix = np.pad(starts_ix,(1,0), mode='constant', constant_values=(1))
#             ends_ix = np.pad(ends_ix,(0,1), mode='constant', constant_values=(a.size + 1))
            
        l0 = ends_ix - starts_ix
        starts_ix1 = np.pad(starts_ix[1:],(0,1), mode='constant', constant_values=(a.size + 1)) 
        l1 = starts_ix1 - (starts_ix + l0)
        for i in range(s1):
            hist[l0[i], l1[i]] += 1
       
    hist[:,0] = 0
    mx = np.max(hist)
    ind = np.where(hist == mx)
    return ind[0][0], ind[1][0]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            images.append(img)
    return images