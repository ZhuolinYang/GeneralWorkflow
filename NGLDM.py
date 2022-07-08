# Neighbouring Grey Level Dependence Based Freatures

'''
Following the IBSI, there are 17 features in this category.
Require image quantisation (default 16).
'''

import numpy as np
import cv2

def Quantisation(ima, quantise = 16):
    min = np.nanmin(ima)
    max = np.nanmax(ima)
    ima_quantised = np.trunc((ima-min) / ((max - min) / quantise)) + 1
    ima_quantised[np.where(ima_quantised == np.max(ima_quantised))] = quantise
    return ima_quantised

def get_NGLDMmatrix(ima):
    ima_quantised = Quantisation(ima)
    Ng = np.max(ima_quantised)
    Nn = 9 # number of the neighbourhoods + 1
    ima_pad = cv2.copyMakeBorder(ima_quantised, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    kernel_h, kernel_w = kernel.shape
    NGLDM = np.zeros((int(Ng), int(Nn)))
    for i in range(ima_quantised.shape[1]):
        for j in range(ima_quantised.shape[0]):
            multiply = ima_pad[i: i + kernel_h, j: j + kernel_w] * kernel
            row = int(ima_quantised[i, j])
            col = int(np.sum(multiply == row))
            NGLDM[row-1][col] += 1
    # delete columes with all zeros
    idx = np.argwhere(np.all(NGLDM[..., :] == 0, axis=0))
    NGLDMmatrix = np.delete(NGLDM, idx, axis=1)
    return NGLDMmatrix


def get_LowDependenceEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_j = np.sum(NGLDMmatrix, axis=0)
    j = np.array(range(1, NGLDMmatrix.shape[1] + 1))
    return np.sum(s_j / (j ** 2)) / Ns


def get_HighDependenceEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_j = np.sum(NGLDMmatrix, axis=0)
    j = np.array(range(1, NGLDMmatrix.shape[1] + 1))
    return np.sum(s_j * (j ** 2)) / Ns


def get_LowGreyLevelCountEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_i = np.sum(NGLDMmatrix, axis=1)
    i = np.array(range(1, NGLDMmatrix.shape[0] + 1))
    return np.sum(s_i / (i ** 2)) / Ns


def get_HighGreyLevelCountEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_i = np.sum(NGLDMmatrix, axis=1)
    i = np.array(range(1, NGLDMmatrix.shape[0] + 1))
    return np.sum(s_i * (i ** 2)) / Ns


def get_LowDependenceLowGreyLevelEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    col, row = np.meshgrid(range(1, NGLDMmatrix.shape[1] + 1), range(1, NGLDMmatrix.shape[0] + 1))
    return np.sum(NGLDMmatrix / ((row ** 2) * (col ** 2))) / Ns


def get_LowDependenceHighGreyLevelEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    col, row = np.meshgrid(range(1, NGLDMmatrix.shape[1] + 1), range(1, NGLDMmatrix.shape[0] + 1))
    return np.sum(NGLDMmatrix * (row ** 2) / (col ** 2)) / Ns


def get_HighDependenceLowGreyLevelEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    col, row = np.meshgrid(range(1, NGLDMmatrix.shape[1] + 1), range(1, NGLDMmatrix.shape[0] + 1))
    return np.sum(NGLDMmatrix * (col ** 2) / (row ** 2)) / Ns


def get_HighDependenceHighGreyLevelEmphasis(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    col, row = np.meshgrid(range(1, NGLDMmatrix.shape[1] + 1), range(1, NGLDMmatrix.shape[0] + 1))
    return np.sum(NGLDMmatrix * (col ** 2) * (row ** 2)) / Ns


def get_GreyLevelNonUniformity(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_i = np.sum(NGLDMmatrix, axis=1)
    return np.sum(s_i ** 2) / Ns


def get_NormalisedGreyLevelNonUniformity(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_i = np.sum(NGLDMmatrix, axis=1)
    return np.sum(s_i ** 2) / (Ns ** 2)


def get_DependeceCountNonUniformity(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_j = np.sum(NGLDMmatrix, axis=0)
    return np.sum(s_j ** 2) / Ns


def get_NormalisedDependeceCountNonUniformity(NGLDMmatrix):
    Ns = np.sum(NGLDMmatrix)
    s_j = np.sum(NGLDMmatrix, axis=0)
    return np.sum(s_j ** 2) / (Ns ** 2)


def get_DependenceCountPercentage(ima, NGLDMmatrix):
    # The feature is equal to one in IBSI definition
    Ns = np.sum(NGLDMmatrix)
    Nv = ima.size
    return Ns / Nv


def get_GreyLevelVariance(NGLDMmatrix):
    p_ij = NGLDMmatrix / np.sum(NGLDMmatrix)
    col, row = np.meshgrid(range(1, NGLDMmatrix.shape[1] + 1), range(1, NGLDMmatrix.shape[0] + 1))
    mu = np.sum(row * p_ij)
    return np.sum(((row - mu) ** 2) * p_ij)


def get_DependenceCountVariance(NGLDMmatrix):
    p_ij = NGLDMmatrix / np.sum(NGLDMmatrix)
    col, row = np.meshgrid(range(1, NGLDMmatrix.shape[1] + 1), range(1, NGLDMmatrix.shape[0] + 1))
    mu = np.sum(col * p_ij)
    return np.sum(((col - mu) ** 2) * p_ij)


def get_DependenceCountEntropy(NGLDMmatrix):
    p_ij = NGLDMmatrix / np.sum(NGLDMmatrix)
    return -1 * np.sum(p_ij * np.log2(p_ij + 1e-7))  # plus a small number to avoid -Inf


def get_DependenceCountEnergy(NGLDMmatrix):
    p_ij = NGLDMmatrix / np.sum(NGLDMmatrix)
    return np.sum(p_ij ** 2)

def get_NGLDMfeatures(ima):
    NGLDM = get_NGLDMmatrix(ima)

    Low_Dependence_Emphasis = get_LowDependenceEmphasis(NGLDM)
    High_Dependence_Emphasis = get_HighDependenceEmphasis(NGLDM)
    Low_Grey_Level_Count_Emphasis = get_LowGreyLevelCountEmphasis(NGLDM)
    High_Grey_Level_Count_Emphasis = get_HighGreyLevelCountEmphasis(NGLDM)
    Low_Dependence_Low_Grey_Level_Emphasis = get_LowDependenceLowGreyLevelEmphasis(NGLDM)
    Low_Dependence_High_Grey_Level_Emphasis = get_LowDependenceHighGreyLevelEmphasis(NGLDM)
    High_Dependence_Low_Grey_Level_Emphasis = get_HighDependenceLowGreyLevelEmphasis(NGLDM)
    High_Dependence_High_Grey_Level_Emphasis = get_HighDependenceHighGreyLevelEmphasis(NGLDM)
    Grey_Level_NonUniformity = get_GreyLevelNonUniformity(NGLDM)
    Normalised_Grey_Level_NonUniformity = get_NormalisedGreyLevelNonUniformity(NGLDM)
    Dependence_Count_NonUniformity = get_DependeceCountNonUniformity(NGLDM)
    Normalised_Dependence_Count_NonUniformity = get_NormalisedDependeceCountNonUniformity(NGLDM)
    Dependence_Count_Percentage = get_DependenceCountPercentage(ima, NGLDM)
    Grey_Level_Variance = get_GreyLevelVariance(NGLDM)
    Dependence_Count_Variance = get_DependenceCountVariance(NGLDM)
    Dependence_Count_Entropy = get_DependenceCountEntropy(NGLDM)
    Dependence_Count_Energy = get_DependenceCountEnergy(NGLDM)

    return Low_Dependence_Emphasis, High_Dependence_Emphasis, Low_Grey_Level_Count_Emphasis,\
           High_Grey_Level_Count_Emphasis, Low_Dependence_Low_Grey_Level_Emphasis,\
           Low_Dependence_High_Grey_Level_Emphasis, High_Dependence_Low_Grey_Level_Emphasis,\
           High_Dependence_High_Grey_Level_Emphasis, Grey_Level_NonUniformity,\
           Normalised_Grey_Level_NonUniformity,Dependence_Count_NonUniformity,\
           Normalised_Dependence_Count_NonUniformity, Dependence_Count_Percentage,\
           Grey_Level_Variance, Dependence_Count_Variance, Dependence_Count_Entropy,\
           Dependence_Count_Energy

