# Neighbouring Grey Tone Difference Based Freatures

'''
Following the IBSI, there are 5  features in this category.
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


def get_NGTDMmatrix(ima):
    ima_quantised = Quantisation(ima)
    # obtain the average matrix through convolution
    ima_pad = cv2.copyMakeBorder(ima_quantised, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
    ima_h, ima_w = ima_pad.shape
    kernel_h, kernel_w = kernel.shape
    # size of the average matrix
    ave_h = ima_h - kernel_h + 1
    ave_w = ima_w - kernel_w + 1
    ave = np.zeros((ave_h, ave_w), dtype = np.float)
    # convolution
    for i in range(ave_h):
        for j in range(ave_w):
            multiply = ima_pad[i : i+kernel_h, j : j+kernel_w] * kernel
            n = np.count_nonzero(multiply) # the number of non-zero elements
            ave[i][j] = np.sum(multiply) / n
    # NGTDM
    Ng = np.max(ima_quantised)
    NGTDM = np.zeros(int(Ng))
    for i in range(int(Ng)):
        index = np.argwhere(ima_quantised == i+1)
        for j in range(len(index)):
            NGTDM[i] += ave[index[j][0], index[j][1]]

    return NGTDM


def get_hist(ima):
    Nbins = np.max(ima) - np.min(ima) + 1
    hist,_ = np.histogram(ima.flatten(), bins = int(Nbins))
    p_i = hist / ima.size # occurrence probability
    return hist, p_i


def get_Coarseness(ima):
    s_i = get_NGTDMmatrix(ima) # get_NGTDMmatrix has already included quantisation
    ima_quantised = Quantisation(ima)
    n_i, p_i = get_hist(ima_quantised)
    den = np.sum(p_i * s_i)
    if den == 0:
        Coarseness = 10 ** 6  # IBSI rule
    else:
        Coarseness = 1 / den
    return Coarseness


def get_Contrast(ima):
    s_i = get_NGTDMmatrix(ima)  # get_NGTDMmatrix has already included quantisation
    ima_quantised = Quantisation(ima)
    n_i, p_i = get_hist(ima_quantised)
    N_gp = np.count_nonzero(p_i)
    N_vc = ima.size
    if N_gp == 1:
        Contrast = 0
    else:
        i2, i1 = np.meshgrid(range(1, len(p_i) + 1), range(1, len(p_i) + 1))
        i1 = i1.flatten()
        i2 = i2.flatten()
        p_i1 = np.repeat(p_i, len(p_i))
        p_i2 = np.tile(p_i, len(p_i))
        Contrast =  (np.sum(p_i1 * p_i2 * ((i1 - i2) ** 2)) / (N_gp * (N_gp - 1))) * (np.sum(s_i) / N_vc)
    return Contrast


def get_Busyness(ima):
    s_i = get_NGTDMmatrix(ima)  # get_NGTDMmatrix has already included quantisation
    ima_quantised = Quantisation(ima)
    n_i, p_i = get_hist(ima_quantised)
    N_gp = np.count_nonzero(p_i)
    if N_gp == 1:
        Busyness = 0
    else:
        i2, i1 = np.meshgrid(range(1, len(p_i) + 1), range(1, len(p_i) + 1))
        i1 = i1.flatten()
        i2 = i2.flatten()
        p_i1 = np.repeat(p_i, len(p_i))
        p_i2 = np.tile(p_i, len(p_i))
        res = np.zeros(len(i1))
        num = np.sum(p_i * s_i)
        for i in range(len(i1)):
            if (p_i1[i] != 0 and p_i2[i] != 0):
                res[i] = np.abs(i1[i] * p_i1[i] - i2[i] * p_i2[i])
        den = np.sum(res)
        Busyness = num / den
    return Busyness


def get_Complexity(ima):
    s_i = get_NGTDMmatrix(ima)  # get_NGTDMmatrix has already included quantisation
    ima_quantised = Quantisation(ima)
    n_i, p_i = get_hist(ima_quantised)
    N_vc = ima.size
    i2, i1 = np.meshgrid(range(1, len(p_i) + 1), range(1, len(p_i) + 1))
    i1 = i1.flatten()
    i2 = i2.flatten()
    p_i1 = np.repeat(p_i, len(p_i))
    p_i2 = np.tile(p_i, len(p_i))
    s_i1 = np.repeat(s_i, len(s_i))
    s_i2 = np.tile(s_i, len(s_i))
    res = np.zeros(len(i1))
    for i in range(len(i1)):
        if (p_i1[i] != 0 and p_i2[i] != 0):
            res[i] = np.abs(i1[i] - i2[i]) * (p_i1[i] * s_i1[i] + p_i2[i] * s_i2[i]) / (p_i1[i] + p_i2[i])
    Complexity = np.sum(res) / N_vc
    return Complexity


def get_Strength(ima):
    s_i = get_NGTDMmatrix(ima)  # get_NGTDMmatrix has already included quantisation
    ima_quantised = Quantisation(ima)
    n_i, p_i = get_hist(ima_quantised)
    den = np.sum(s_i)
    if den == 0:
        Strength = 0
    else:
        i2, i1 = np.meshgrid(range(1, len(p_i) + 1), range(1, len(p_i) + 1))
        i1 = i1.flatten()
        i2 = i2.flatten()
        p_i1 = np.repeat(p_i, len(p_i))
        p_i2 = np.tile(p_i, len(p_i))
        res = np.zeros(len(i1))
        for i in range(len(i1)):
            if (p_i1[i] != 0 and p_i2[i] != 0):
                res[i] = (p_i1[i] + p_i2[i]) * ((i1[i] - i2[i]) ** 2)
        num = np.sum(res)
        Strength = num / den
    return Strength


def get_NGTDMfeatures(ima):
    Coarseness = get_Coarseness(ima)
    Contrast = get_Contrast(ima)
    Busyness = get_Busyness(ima)
    Complexity = get_Complexity(ima)
    Strength = get_Strength(ima)
    return Coarseness, Contrast, Busyness, Complexity, Strength