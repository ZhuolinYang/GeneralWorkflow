# Grey Level Distance Zone Based Features

'''
Following the IBSI, there are 16 features in this category.
Require image quantisation (default 16).

In this 2D approach, we consider 4 connectedness.

Ng: number of grey levels in the quantised image (default 16)
Nd: the largest distance of any zone
Nv: number of pixels in the input image
NS: total zone count
d_i: the number of zones with discretised grey level 𝑖, regardless of distance
d_j: the number of zones with distance 𝑗, regardless of grey level
'''

import numpy as np
import cv2
import math
from skimage.measure import label

def Quantisation(ima, quantise = 16):
    min = np.nanmin(ima)
    max = np.nanmax(ima)
    ima_quantised = np.trunc((ima-min) / ((max - min) / quantise)) + 1
    ima_quantised[np.where(ima_quantised == np.max(ima_quantised))] = quantise
    return ima_quantised


def get_GLDZMmatrix(ima):
    ima_quantised = Quantisation(ima)
    m, n = ima_quantised.shape
    Ng = np.max(ima_quantised)
    Nd = math.ceil(min(m, n) / 2)
    GLDZM = np.zeros((int(Ng), int(Nd)))
    # generate dist_map
    mask = np.ones_like(ima_quantised)
    # in order to get the distance map by eroding the mask, we have to pad the mask first
    mask_pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # convert the data type, otherwise, erosion will cause errors
    erosion = mask_pad.astype('uint8')
    kernel = np.ones((3,3), dtype = np.uint8)
    # erode the ROI mask until the mask is empty (all 0s no 1s)
    i = 0
    while i < math.ceil(max(m,n) / 2) - 1:
        erosion = cv2.erode(erosion, kernel, iterations=1)
        index = np.argwhere(erosion == 1)
        for j in range(len(index)):
            mask[int(index[j][0] - 1), int(index[j][1] - 1)] += 1
        i += 1
    dist_map = mask
    # generate zone matirx with 4-connectedness
    labels = label(ima_quantised, connectivity = 1)
    zones = np.max(labels)
    # generate GLDZM using dist_map and zone matrix
    for i in range(zones):
        index = np.argwhere(labels == i+1)
        row = int(ima_quantised[int(index[0][0]), int(index[0][1])])
        distance = []
        for j in range(len(index)):
            dist = dist_map[int(index[j][0]), int(index[j][1])]
            distance.append(dist)
        col = int(min(distance))
        GLDZM[row-1][col-1] += 1
    # delete columes with all zeros
    idx = np.argwhere(np.all(GLDZM[..., :] == 0, axis=0))
    GLDZMmatrix = np.delete(GLDZM, idx, axis=1)
    return GLDZMmatrix


def get_SmallDistanceEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_j = np.sum(GLDZMmatrix, axis=0)
    j = np.array(range(1, GLDZMmatrix.shape[1] + 1))
    return np.sum(d_j / (j ** 2)) / Ns


def get_LargeDistanceEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_j = np.sum(GLDZMmatrix, axis=0)
    j = np.array(range(1, GLDZMmatrix.shape[1] + 1))
    return np.sum(d_j * (j ** 2)) / Ns


def get_LowGreyLevelZoneEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_i = np.sum(GLDZMmatrix, axis=1)
    i = np.array(range(1, GLDZMmatrix.shape[0] + 1))
    return np.sum(d_i / (i ** 2)) / Ns


def get_HighGreyLevelZoneEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_i = np.sum(GLDZMmatrix, axis=1)
    i = np.array(range(1, GLDZMmatrix.shape[0] + 1))
    return np.sum(d_i * (i ** 2)) / Ns


def get_SmallDistanceLowGreyLevelEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    col, row = np.meshgrid(range(1, GLDZMmatrix.shape[1] + 1), range(1, GLDZMmatrix.shape[0] + 1))
    return np.sum(GLDZMmatrix / ((row ** 2) * (col ** 2))) / Ns


def get_SmallDistanceHighGreyLevelEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    col, row = np.meshgrid(range(1, GLDZMmatrix.shape[1] + 1), range(1, GLDZMmatrix.shape[0] + 1))
    return np.sum(GLDZMmatrix * (row ** 2) / (col ** 2)) / Ns


def get_LargeDistanceLowGreyLevelEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    col, row = np.meshgrid(range(1, GLDZMmatrix.shape[1] + 1), range(1, GLDZMmatrix.shape[0] + 1))
    return np.sum(GLDZMmatrix * (col ** 2) / (row ** 2)) / Ns


def get_LargeDistanceHighGreyLevelEmphasis(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    col, row = np.meshgrid(range(1, GLDZMmatrix.shape[1] + 1), range(1, GLDZMmatrix.shape[0] + 1))
    return np.sum(GLDZMmatrix * (col ** 2) * (row ** 2)) / Ns


def get_GreyLevelNonUniformity(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_i = np.sum(GLDZMmatrix, axis=1)
    return np.sum(d_i ** 2) / Ns


def get_NormalisedGreyLevelNonUniformity(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_i = np.sum(GLDZMmatrix, axis=1)
    return np.sum(d_i ** 2) / (Ns ** 2)


def get_ZoneDistanceNonUniformity(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_j = np.sum(GLDZMmatrix, axis=0)
    return np.sum(d_j ** 2) / Ns


def get_NormalisedZoneDistanceNonUniformity(GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    d_j = np.sum(GLDZMmatrix, axis=0)
    return np.sum(d_j ** 2) / (Ns ** 2)


def get_ZonePercentage(ima, GLDZMmatrix):
    Ns = np.sum(GLDZMmatrix)
    Nv = ima.size
    return Ns / Nv


def get_GreyLevelVariance(GLDZMmatrix):
    p_ij = GLDZMmatrix / np.sum(GLDZMmatrix)
    col, row = np.meshgrid(range(1, GLDZMmatrix.shape[1] + 1), range(1, GLDZMmatrix.shape[0] + 1))
    mu = np.sum(row * p_ij)
    return np.sum(((row - mu) ** 2) * p_ij)


def get_ZoneDistanceVariance(GLDZMmatrix):
    p_ij = GLDZMmatrix / np.sum(GLDZMmatrix)
    col, row = np.meshgrid(range(1, GLDZMmatrix.shape[1] + 1), range(1, GLDZMmatrix.shape[0] + 1))
    mu = np.sum(col * p_ij)
    return np.sum(((col - mu) ** 2) * p_ij)


def get_ZoneDistanceEntropy(GLDZMmatrix):
    p_ij = GLDZMmatrix / np.sum(GLDZMmatrix)
    return -1 * np.sum(p_ij * np.log2(p_ij + 1e-7)) # plus a small number to avoid -Inf


def get_GLDZMfeatures(ima):
    GLDZM = get_GLDZMmatrix(ima)

    Small_Distance_Emphasis = get_SmallDistanceEmphasis(GLDZM)
    Large_Distance_Emphasis = get_LargeDistanceEmphasis(GLDZM)
    Low_Grey_Level_Zone_Emphasis = get_LowGreyLevelZoneEmphasis(GLDZM)
    High_Grey_Level_Zone_Emphasis = get_HighGreyLevelZoneEmphasis(GLDZM)
    Small_Distance_Low_Grey_Level_Emphasis = get_SmallDistanceLowGreyLevelEmphasis(GLDZM)
    Small_Distance_High_Grey_Level_Emphasis = get_SmallDistanceHighGreyLevelEmphasis(GLDZM)
    Large_Distance_Low_Grey_Level_Emphasis = get_LargeDistanceLowGreyLevelEmphasis(GLDZM)
    Large_Distance_High_Grey_Level_Emphasis = get_LargeDistanceHighGreyLevelEmphasis(GLDZM)
    Grey_Level_NonUniformity = get_GreyLevelNonUniformity(GLDZM)
    Normalised_Grey_Level_NonUniformity = get_NormalisedGreyLevelNonUniformity(GLDZM)
    Zone_Distance_NonUniformity = get_ZoneDistanceNonUniformity(GLDZM)
    Normalised_Zone_Distance_NonUniformity = get_NormalisedZoneDistanceNonUniformity(GLDZM)
    Zone_Percentage = get_ZonePercentage(ima, GLDZM)
    Grey_Level_Variance = get_GreyLevelVariance(GLDZM)
    Zone_Distance_Variance = get_ZoneDistanceVariance(GLDZM)
    Zone_Distance_Entropy = get_ZoneDistanceEntropy(GLDZM)

    return Small_Distance_Emphasis, Large_Distance_Emphasis, Low_Grey_Level_Zone_Emphasis,\
           High_Grey_Level_Zone_Emphasis, Small_Distance_Low_Grey_Level_Emphasis,\
           Small_Distance_High_Grey_Level_Emphasis, Large_Distance_Low_Grey_Level_Emphasis,\
           Large_Distance_High_Grey_Level_Emphasis, Grey_Level_NonUniformity,\
           Normalised_Grey_Level_NonUniformity, Zone_Distance_NonUniformity,\
           Normalised_Zone_Distance_NonUniformity, Zone_Percentage, Grey_Level_Variance,\
           Zone_Distance_Variance, Zone_Distance_Entropy

















