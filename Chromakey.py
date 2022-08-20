from turtle import color, width
import cv2 as cv
import numpy as np
import sys
'''
Student Name: Yuke Li
Student Number:7357394
'''
def img_resize(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        dim = (w, h)
    elif width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    resized = cv.resize(image, dim, interpolation=cv.INTER_LINEAR)

    return resized

# Color space convert
class ColorSplit:
    def __init__(self, img):
        self.img = img
        self.cvt_img = np.zeros(shape= self.img.shape, dtype='uint8')
        self.img_width = 540 # 1280*720

# 2 XYZ
    def cvtXYZ(self):
        self.cvt_img = cv.cvtColor(self.img, cv.COLOR_BGR2XYZ)
# 2 Lab
    def cvtLab(self):
        self.cvt_img = cv.cvtColor(self.img, cv.COLOR_BGR2LAB)
# 2 YCrCb
    def cvtYcrcb(self):
        self.cvt_img = cv.cvtColor(self.img, cv.COLOR_BGR2YCrCb)  
# 2 Hsv
    def cvtHsv(self):
        self.cvt_img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)

    def ImageDisplay(self):
        #initialization for 3 channels
        channel_1 = np.zeros(shape=self.cvt_img.shape, dtype='uint8')
        channel_2 = np.zeros(shape=self.cvt_img.shape, dtype='uint8')
        channel_3 = np.zeros(shape=self.cvt_img.shape, dtype='uint8')
    
        # channel 1 to img
        channel_1[:,:,0] = self.cvt_img[:,:,0]
        channel_1[:,:,1] = self.cvt_img[:,:,0]
        channel_1[:,:,2] = self.cvt_img[:,:,0]

        # channel 2 to img
        channel_2[:,:,0] = self.cvt_img[:,:,0]
        channel_2[:,:,1] = self.cvt_img[:,:,0]
        channel_2[:,:,2] = self.cvt_img[:,:,0]

        # channel 3 to img
        channel_3[:,:,0] = self.cvt_img[:,:,0]
        channel_3[:,:,1] = self.cvt_img[:,:,0]
        channel_3[:,:,2] = self.cvt_img[:,:,0]

        # resizing
        ini_img = img_resize(self.img, width=self.img_width)
        channel_1 = img_resize(channel_1, width=self.img_width)
        channel_2 = img_resize(channel_2, width=self.img_width)
        channel_3 = img_resize(channel_3, width=self.img_width)

        # layout
        img_top = np.hstack((ini_img,channel_1))
        img_bottom = np.hstack((channel_2,channel_3))
        dispaly = np.vstack((img_top,img_bottom))

        return dispaly

class Blend:
    def __init__(self, green_img, back_img):
        self.green_img = green_img
        self.back_img = back_img
        self.img_width = 540 # 1280*720
        
        # green img process hsv 
        self.green_img_hsv = cv.cvtColor(self.green_img, cv.COLOR_BGR2HSV)

        # senic picture process
        self.back_img = cv.resize(self.back_img, (self.green_img.shape[1], self.green_img.shape[0]))

        #initialization 
        self.mask = np.zeros(shape=self.green_img.shape, dtype='uint8')
        self.front = np.zeros(shape=self.green_img.shape, dtype='uint8')
        self.back = np.zeros(shape=self.green_img.shape, dtype='uint8')
        self.final = np.zeros(shape=self.green_img.shape, dtype='uint8')

    def mask_extract(self):

        # lower and upper green (tesitng.py, setTrackbarPos,createTrackbar)
        l_green = (32, 82, 81)
        u_green = (152, 255, 255)

        mask = cv.inRange(self.green_img_hsv, l_green, u_green)
        mask = 255 - mask

        # apply morphology opening to mask
        kernel = np.ones((1,1), np.uint8)
        # erode edge of mask, black = 0 (back)
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)
        # remove the back points in front
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        self.mask = cv.GaussianBlur(mask,(1,1),sigmaX=1,sigmaY=1, borderType=cv.BORDER_DEFAULT) 
        #mask green = 0
    
    def gene_front(self):
        self.front = cv.bitwise_and(self.green_img,self.green_img,mask=self.mask)

    def gene_back(self):
        mask = 255 - self.mask
        self.back = cv.bitwise_and(self.back_img, self.back_img, mask=mask)
    
    def belend_img(self):
        self.final = cv.bitwise_or(self.front, self.back)

    def display(self):
        channel_1 = self.green_img.copy()
        channel_1[self.mask==0] = (255, 255, 255)

        with_green = img_resize(self.green_img, width = self.img_width)
        with_white = img_resize(channel_1, width = self.img_width)
        back_img = img_resize(self.back_img, width = self.img_width)
        final = img_resize(self.final, width = self.img_width)

        img_top = np.hstack((with_green, with_white))
        img_bottom = np.hstack((back_img,final))
        display = np.vstack((img_top,img_bottom))

        return display 

def split(color_space, path):
    img = cv.imread(path)
    color_converted = ColorSplit(img)
    if color_space == '-XYZ':
        color_converted.cvtXYZ()
    elif color_space == '-Lab':
        color_converted.cvtLab()
    elif color_space == '-YCrCb':
        color_converted.cvtYcrcb()
    elif color_space == '-HSB':
        color_converted.cvtHsv()
    
    display_img = color_converted.ImageDisplay()

    cv.imshow('Color Space {}'.format(color_space), display_img)
    cv.waitKey(0)

def green_screen(green_path, back_path):
    green_img = cv.imread(green_path)
    back_img = cv.imread(back_path)

    blend_back = Blend(green_img,back_img)
    # process blend img
    blend_back.mask_extract()
    blend_back.gene_front()
    blend_back.gene_back()
    blend_back.belend_img()

    display_img = blend_back.display()
    cv.imshow('Green screen masking', display_img)
    cv.waitKey(0)

args = sys.argv
para = args[1]
img = args[2]

if (len(para.rsplit('.',1))) > 1 :
    green_screen(img, para)
else:
    split(para, img)
