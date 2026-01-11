import numpy as np
import cv2
from feature_moments import getShapeFeatures
from feature_gabor import *
from feature_color import getColorFeature
from img_seg import *

def createFeature(img):
    feature = []
    areaFruit, binaryImg, colourImg, areaSkin, fruitContour, pix_to_cm_multiplier = getAreaOfFood(img)
    color = getColorFeature(colourImg)
    texture = getTextureFeature(colourImg)
    shape = getShapeFeatures(binaryImg)
    
    for i in color:
        feature.append(i)
    for i in texture:
        feature.append(i)
    for i in shape:
        feature.append(i)
    
    # تحويل القائمة لمصفوفة numpy لتسهيل العمليات الحسابية
    feature = np.array(feature)
    M = max(feature)
    m = min(feature)
    
    if M != m:
        feature = (feature - m) / (M - m)
    
    mean = np.mean(feature)
    dev = np.std(feature)
    if dev != 0:
        feature = (feature - mean) / dev
        
    return feature, areaFruit, areaSkin, fruitContour, pix_to_cm_multiplier
