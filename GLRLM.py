# GLRLM-based Features
'''
Following the IBSI, there are 16 features in this category.
Require image quantisation (default 16).

GLRLM-based features are calculated from a single matrix after merging 4 2D directional metirces (0 degree, 45 degree,
90 degree, 135 degree with step size = 1).

Ng: number of grey levels of the quantised image (default 16)
Nr: maximal possible run length
Nv: Nv: number of pixels in the input image, following IBSI, since the matrices are merged by summing the run counts
of each matrix element (𝑖,𝑗) over the different matrices, Nv should likewise be summed to retain consistency
Ns: the sum over all elements in the GLRLM matrix
r_i, r_j: marginal sums
'''

import numpy as np
from itertools import groupby

def Quantisation(ima, quantise = 16):
    min = np.nanmin(ima)
    max = np.nanmax(ima)
    ima_quantised = np.trunc((ima-min) / ((max - min) / quantise)) + 1
    ima_quantised[np.where(ima_quantised == np.max(ima_quantised))] = quantise
    return ima_quantised


def get_GLRLMmatrix(ima):
    ima_quantised = Quantisation(ima)
    m, n = ima_quantised.shape
    run_length = max(m, n)
    grey_levels = np.nanmax(ima_quantised) - np.nanmin(ima_quantised) + 1

    G0 = [val.tolist() for sublist in np.vsplit(ima_quantised, m) for val in sublist]
    diag45 = [ima_quantised[::-1, :].diagonal(i) for i in range(-ima_quantised.shape[0] + 1, ima_quantised.shape[1])]
    G45 = [n.tolist() for n in diag45]
    G90 = [val.tolist() for sublist in np.split(np.transpose(ima_quantised), n) for val in sublist]
    ima_trans = np.rot90(ima_quantised, 3)
    diag135 = [ima_trans[::-1, :].diagonal(i) for i in range(-ima_trans.shape[0] + 1, ima_trans.shape[1])]
    G135 = [n.tolist() for n in diag135]

    def length(l):
        if hasattr(l, '__len__'):
            return np.size(l)
        else:
            i = 0
            for _ in l:
                i += 1
            return i

    theta = ['G0', 'G45', 'G90', 'G135']

    GLRLMs = np.zeros((4, int(grey_levels), int(run_length))) # 4 means there are 4 directions
    for angle in theta:
        for splitvec in range(0, len(eval(angle))):
            flattened = eval(angle)[splitvec]
            result = []
            for key, iter in groupby(flattened):
                result.append((key, length(iter)))
            for resIndex in range(0, len(result)):
                GLRLMs[theta.index(angle), int(result[resIndex][0] - np.nanmin(ima_quantised)), int(result[resIndex][1] - 1)] += 1

    GLRLM = (GLRLMs[0] + GLRLMs[1] + GLRLMs[2] + GLRLMs[3])

    return GLRLM


def get_ShortRunsEmphasis(GLRLMmatrix):
    j = np.array(range(1, GLRLMmatrix.shape[1]+1))
    Ns = np.sum(GLRLMmatrix)
    r_j = np.sum(GLRLMmatrix, axis = 0)
    return np.sum(r_j / (j ** 2)) / Ns


def get_LongRunsEmphasis(GLRLMmatrix):
    j = np.array(range(1, GLRLMmatrix.shape[1] + 1))
    Ns = np.sum(GLRLMmatrix)
    r_j = np.sum(GLRLMmatrix, axis=0)
    return np.sum(r_j * (j ** 2)) / Ns


def get_LowGreyLevelRunEmphasis(GLRLMmatrix):
    i = np.array(range(1, GLRLMmatrix.shape[0] + 1))
    Ns = np.sum(GLRLMmatrix)
    r_i = np.sum(GLRLMmatrix, axis=1)
    return np.sum(r_i / (i ** 2)) / Ns


def get_HighGreyLevelRunEmphasis(GLRLMmatrix):
    i = np.array(range(1, GLRLMmatrix.shape[0] + 1))
    Ns = np.sum(GLRLMmatrix)
    r_i = np.sum(GLRLMmatrix, axis=1)
    return np.sum(r_i * (i ** 2)) / Ns


def get_ShortRunLowGreyLevelEmphasis(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    col, row = np.meshgrid(range(1, GLRLMmatrix.shape[1] + 1), range(1, GLRLMmatrix.shape[0] + 1))
    return np.sum(GLRLMmatrix / ((row**2) * (col**2))) / Ns


def get_ShortRunHighGreyLevelEmphasis(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    col, row = np.meshgrid(range(1, GLRLMmatrix.shape[1] + 1), range(1, GLRLMmatrix.shape[0] + 1))
    return np.sum(GLRLMmatrix * (row ** 2) / (col ** 2)) / Ns


def get_LongRunLowGreyLevelEmphasis(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    col, row = np.meshgrid(range(1, GLRLMmatrix.shape[1] + 1), range(1, GLRLMmatrix.shape[0] + 1))
    return np.sum(GLRLMmatrix * (col ** 2) / (row ** 2)) / Ns


def get_LongRunHighGreyLevelEmphasis(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    col, row = np.meshgrid(range(1, GLRLMmatrix.shape[1] + 1), range(1, GLRLMmatrix.shape[0] + 1))
    return np.sum(GLRLMmatrix * (col ** 2) * (row ** 2)) / Ns


def get_GreyLevelNonUniformity(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    r_i = np.sum(GLRLMmatrix, axis=1)
    return np.sum(r_i ** 2) / Ns


def get_NormalisedGreyLevelNonUniformity(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    r_i = np.sum(GLRLMmatrix, axis=1)
    return np.sum(r_i ** 2) / (Ns ** 2)


def get_RunLengthNonUniformity(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    r_j = np.sum(GLRLMmatrix, axis=1)
    return np.sum(r_j ** 2) / Ns


def get_NormalisedRunLengthNonUniformity(GLRLMmatrix):
    Ns = np.sum(GLRLMmatrix)
    r_j = np.sum(GLRLMmatrix, axis=1)
    return np.sum(r_j ** 2) / (Ns ** 2)


def get_RunPercentage(ima, GLRLMmatrix):
    # Following the IBSI, when this feature is calculated using a merged GLRLM,
    # the denominator should be the number of voxels of the underlying matrices
    # 4 means we merged 4 metirces including G0, G45, G90, G135
    Ns = np.sum(GLRLMmatrix)
    Nv = 4 * ima.shape[0] * ima.shape[1]
    return Ns / Nv


def get_GreyLevelVariance(GLRLMmatrix):
    p_ij = GLRLMmatrix / np.sum(GLRLMmatrix)
    col, row = np.meshgrid(range(1, GLRLMmatrix.shape[1] + 1), range(1, GLRLMmatrix.shape[0] + 1))
    mu = np.sum(row * p_ij)
    return np.sum(((row - mu) ** 2) * p_ij)


def get_RunLengthVariance(GLRLMmatrix):
    p_ij = GLRLMmatrix / np.sum(GLRLMmatrix)
    col, row = np.meshgrid(range(1, GLRLMmatrix.shape[1] + 1), range(1, GLRLMmatrix.shape[0] + 1))
    mu = np.sum(col * p_ij)
    return np.sum(((col - mu) ** 2) * p_ij)


def get_RunEntropy(GLRLMmatrix):
    p_ij = GLRLMmatrix / np.sum(GLRLMmatrix)
    return -1 * np.sum(p_ij * np.log2(p_ij + 1e-7)) # plus a small number to avoid -Inf


def get_GLRLMfeatures(ima):
    '''
    In this function, we calculate GLRLM for 0 degree, 45 degree,
    90 degree and 135 degree
    '''
    GLRLM = get_GLRLMmatrix(ima)

    Short_Runs_Emphasis = get_ShortRunsEmphasis(GLRLM)
    Long_Runs_Emphasis = get_LongRunsEmphasis(GLRLM)
    Low_Grey_Level_Run_Emphasis = get_LowGreyLevelRunEmphasis(GLRLM)
    High_Grey_Level_Run_Emphasis = get_HighGreyLevelRunEmphasis(GLRLM)
    Short_Run_Low_Grey_Level_Emphasis = get_ShortRunLowGreyLevelEmphasis(GLRLM)
    Short_Run_High_Grey_Level_Emphasis = get_ShortRunHighGreyLevelEmphasis(GLRLM)
    Long_Run_Low_Grey_Level_Emphasis = get_LongRunLowGreyLevelEmphasis(GLRLM)
    Long_Run_High_Grey_Level_Emphasis = get_LongRunHighGreyLevelEmphasis(GLRLM)
    Grey_Level_NonUniformity = get_GreyLevelNonUniformity(GLRLM)
    Normalised_Grey_Level_NonUniformity = get_NormalisedGreyLevelNonUniformity(GLRLM)
    Run_Length_NonUniformity = get_RunLengthNonUniformity(GLRLM)
    Normalised_Run_Length_NonUniformity = get_NormalisedRunLengthNonUniformity(GLRLM)
    Run_Percentage = get_RunPercentage(ima, GLRLM)
    Grey_Level_Vriance = get_GreyLevelVariance(GLRLM)
    Run_Length_Variance = get_RunLengthVariance(GLRLM)
    Run_Entropy = get_RunEntropy(GLRLM)

    return Short_Runs_Emphasis, Long_Runs_Emphasis, Low_Grey_Level_Run_Emphasis,\
           High_Grey_Level_Run_Emphasis, Short_Run_Low_Grey_Level_Emphasis,\
           Short_Run_High_Grey_Level_Emphasis, Long_Run_Low_Grey_Level_Emphasis,\
           Long_Run_High_Grey_Level_Emphasis,Grey_Level_NonUniformity,\
           Normalised_Grey_Level_NonUniformity, Run_Length_NonUniformity,\
           Normalised_Run_Length_NonUniformity, Run_Percentage, Grey_Level_Vriance,\
           Run_Length_Variance, Run_Entropy







