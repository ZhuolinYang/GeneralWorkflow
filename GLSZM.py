# Grey Level Size Zone Based Features

'''
Following the IBSI, there are 16 features in this category.
Require image quantisation (default 16).

In this 2D approach, we consider 8 connectedness.

Ng: number of grey levels in the quantised image (default 16)
Nz: maximum zone size of any group of linked pixels
Nv: number of pixels in the input image
NS: total number of zones
s_i, s_j: marginal sums
'''

import numpy as np
from skimage.measure import label

def Quantisation(ima, quantise = 16):
    min = np.nanmin(ima)
    max = np.nanmax(ima)
    ima_quantised = np.trunc((ima-min) / ((max - min) / quantise)) + 1
    ima_quantised[np.where(ima_quantised == np.max(ima_quantised))] = quantise
    return ima_quantised


def get_GLSZMmatrix(ima):
    ima_quantised = Quantisation(ima)
    mask = label(ima_quantised, connectivity = 2)
    mask_hist, _ = np.histogram(mask.flatten(), bins = np.max(mask))
    Nz = np.max(mask_hist)
    Ng = np.max(ima_quantised)
    GLSZM = np.zeros((int(Ng), int(Nz)))
    for i in range(len(mask_hist)):
        col = int(mask_hist[i] - 1)
        index = np.argwhere(mask == i+1)[0]
        row = int(ima_quantised[index[0], index[1]] - 1)
        GLSZM[row][col] += 1

    return GLSZM


def get_SmallZoneEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_j = np.sum(GLSZMmatrix, axis = 0)
    j = np.array(range(1, GLSZMmatrix.shape[1]+1))
    return np.sum(s_j / (j ** 2)) / Ns


def get_LargeZoneEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_j = np.sum(GLSZMmatrix, axis = 0)
    j = np.array(range(1, GLSZMmatrix.shape[1]+1))
    return np.sum(s_j * (j ** 2)) / Ns


def get_LowGreyLevelZoneEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_i = np.sum(GLSZMmatrix, axis=1)
    i = np.array(range(1, GLSZMmatrix.shape[0] + 1))
    return np.sum(s_i / (i ** 2)) / Ns


def get_HighGreyLevelZoneEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_i = np.sum(GLSZMmatrix, axis=1)
    i = np.array(range(1, GLSZMmatrix.shape[0] + 1))
    return np.sum(s_i * (i ** 2)) / Ns


def get_SmallZoneLowGreyLevelEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    col, row = np.meshgrid(range(1, GLSZMmatrix.shape[1] + 1), range(1, GLSZMmatrix.shape[0] + 1))
    return np.sum(GLSZMmatrix / ((row ** 2) * (col ** 2))) / Ns


def get_SmallZoneHighGreyLevelEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    col, row = np.meshgrid(range(1, GLSZMmatrix.shape[1] + 1), range(1, GLSZMmatrix.shape[0] + 1))
    return np.sum(GLSZMmatrix * (row ** 2) / (col ** 2)) / Ns


def get_LargeZoneLowGreyLevelEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    col, row = np.meshgrid(range(1, GLSZMmatrix.shape[1] + 1), range(1, GLSZMmatrix.shape[0] + 1))
    return np.sum(GLSZMmatrix * (col ** 2) / (row ** 2)) / Ns


def get_LargeZoneHighGreyLevelEmphasis(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    col, row = np.meshgrid(range(1, GLSZMmatrix.shape[1] + 1), range(1, GLSZMmatrix.shape[0] + 1))
    return np.sum(GLSZMmatrix * (col ** 2) * (row ** 2)) / Ns


def get_GreyLevelNonUniformity(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_i = np.sum(GLSZMmatrix, axis=1)
    return np.sum(s_i ** 2) / Ns


def get_NormalisedGreyLevelNonUniformity(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_i = np.sum(GLSZMmatrix, axis=1)
    return np.sum(s_i ** 2) / (Ns ** 2)


def get_ZoneSizeNonUniformity(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_j = np.sum(GLSZMmatrix, axis=0)
    return np.sum(s_j ** 2) / Ns


def get_NormalisedZoneSizeNonUniformity(GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    s_j = np.sum(GLSZMmatrix, axis=0)
    return np.sum(s_j ** 2) / (Ns ** 2)


def get_ZonePercentage(ima, GLSZMmatrix):
    Ns = np.sum(GLSZMmatrix)
    Nv = ima.size
    return Ns / Nv


def get_GreyLevelVariance(GLSZMmatrix):
    p_ij = GLSZMmatrix / np.sum(GLSZMmatrix)
    col, row = np.meshgrid(range(1, GLSZMmatrix.shape[1] + 1), range(1, GLSZMmatrix.shape[0] + 1))
    mu = np.sum(row * p_ij)
    return np.sum(((row - mu) ** 2) * p_ij)


def get_ZoneSizeVariance(GLSZMmatrix):
    p_ij = GLSZMmatrix / np.sum(GLSZMmatrix)
    col, row = np.meshgrid(range(1, GLSZMmatrix.shape[1] + 1), range(1, GLSZMmatrix.shape[0] + 1))
    mu = np.sum(col * p_ij)
    return np.sum(((col - mu) ** 2) * p_ij)


def get_ZoneSizeEntropy(GLSZMmatrix):
    p_ij = GLSZMmatrix / np.sum(GLSZMmatrix)
    return -1 * np.sum(p_ij * np.log2(p_ij + 1e-7)) # plus a small number to avoid -Inf


def get_GLSZMfeatures(ima):
    GLSZM = get_GLSZMmatrix(ima)

    Small_Zone_Emphasis = get_SmallZoneEmphasis(GLSZM)
    Large_Zone_Emphasis = get_LargeZoneEmphasis(GLSZM)
    Low_Grey_Level_Zone_Emphasis = get_LowGreyLevelZoneEmphasis(GLSZM)
    High_Grey_Level_Zone_Emphasis = get_HighGreyLevelZoneEmphasis(GLSZM)
    Small_Zone_Low_Grey_Level_Emphasis = get_SmallZoneLowGreyLevelEmphasis(GLSZM)
    Small_Zone_High_Grey_Level_Emphasis = get_SmallZoneHighGreyLevelEmphasis(GLSZM)
    Large_Zone_Low_Grey_Level_Emphasis = get_LargeZoneLowGreyLevelEmphasis(GLSZM)
    Large_Zone_High_Grey_Level_Emphasis = get_LargeZoneHighGreyLevelEmphasis(GLSZM)
    Grey_Level_NonUniformity = get_GreyLevelNonUniformity(GLSZM)
    Normalised_Grey_Level_NonUniformity = get_NormalisedGreyLevelNonUniformity(GLSZM)
    Zone_Size_NonUniformity = get_ZoneSizeNonUniformity(GLSZM)
    Normalised_Zone_Size_NonUniformity = get_NormalisedZoneSizeNonUniformity(GLSZM)
    Zone_Percentage = get_ZonePercentage(ima, GLSZM)
    Grey_Level_Variance = get_GreyLevelVariance(GLSZM)
    Zone_Size_Variance = get_ZoneSizeVariance(GLSZM)
    Zone_Size_Entropy = get_ZoneSizeEntropy(GLSZM)

    return Small_Zone_Emphasis, Large_Zone_Emphasis, Low_Grey_Level_Zone_Emphasis,\
           High_Grey_Level_Zone_Emphasis, Small_Zone_Low_Grey_Level_Emphasis,\
           Small_Zone_High_Grey_Level_Emphasis, Large_Zone_Low_Grey_Level_Emphasis,\
           Large_Zone_High_Grey_Level_Emphasis, Grey_Level_NonUniformity,\
           Normalised_Grey_Level_NonUniformity, Zone_Size_NonUniformity,\
           Normalised_Zone_Size_NonUniformity, Zone_Percentage, Grey_Level_Variance,\
           Zone_Size_Variance, Zone_Size_Entropy


