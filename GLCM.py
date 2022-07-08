# Grey Level Co-occurrence Based Features
'''
Following the IBSI, there are 25 features in this category.
Require image quantisation (default 16).

Matrices are merged by summing the co-occurrence counts in each matrix element (i,j) over the different matrices.
Probability distributions are subsequently calculated for the merged GLCM, which is then used to calculate the GLCM features.

The probability distribution Pm is obtained by normalising Mm by the sum of its elements.
Ng: number of grey levels of the quantised image (default 16).
'''

import numpy as np

def Quantisation(ima, quantise = 16):
    min = np.nanmin(ima)
    max = np.nanmax(ima)
    ima_quantised = np.trunc((ima-min) / ((max - min) / quantise)) + 1
    ima_quantised[np.where(ima_quantised == np.max(ima_quantised))] = quantise
    return ima_quantised


def get_GLCMmatrix(ima, d_x, d_y):
    '''
    Get the symmetric GLCM matrix
    '''
    ima_quantised = Quantisation(ima)
    grey_levels = np.nanmax(ima_quantised) - np.nanmin(ima_quantised) + 1
    m, n = ima_quantised.shape
    
    GLCMmatrix_positive = np.zeros((int(grey_levels), int(grey_levels)))
    if (d_x * d_y) != -1:
        for j in range(m - d_y):
            for i in range(n - d_x):
                rows = int(ima_quantised[j, i])
                cols = int(ima_quantised[j+d_y, i+d_x])
                GLCMmatrix_positive[rows-1, cols-1] += 1.0
    else:
        ima_rot_quantised = np.rot90(ima_quantised)
        for j in range(m - np.abs(d_y)):
            for i in range(n - np.abs(d_x)):
                rows = int(ima_rot_quantised[j, i])
                cols = int(ima_rot_quantised[j + np.abs(d_y), i + np.abs(d_x)])
                GLCMmatrix_positive[rows-1, cols-1] += 1.0

    GLCMmatrix_negative = GLCMmatrix_positive.transpose()

    GLCMmatrix = GLCMmatrix_negative + GLCMmatrix_positive

    return GLCMmatrix


def get_JointMaximum(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    return np.nanmax(p_ij)


def get_JointAverage(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    n = GLCMmatrix.shape[0]
    col, row = np.meshgrid(range(1, p_ij.shape[1]+1), range(1, p_ij.shape[0]+1))
    return np.sum(row * p_ij)


def get_JointVariance(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    mu = get_JointAverage(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    return np.sum(((row - mu) ** 2) * p_ij)


def get_JointEntropy(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    return -1 * np.sum(p_ij * np.log2(p_ij + 1e-7))


def get_DifferenceAvergae(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Diagonal Probability p_IminusJ
    p_IminusJ = np.zeros((1, p_ij.shape[0]))
    p_IminusJ[0][0] = np.sum(np.diagonal(p_ij))
    for i in range(1,p_ij.shape[0]):
        p_IminusJ[0][i] = np.sum(np.diag(p_ij,i)) * 2 # * 2 because of symmetry

    k = np.array(range(0, p_ij.shape[0]))
    return np.sum(p_IminusJ * k)


def get_DifferenceVariance(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Diagonal Probability p_IminusJ
    p_IminusJ = np.zeros((1, p_ij.shape[0]))
    p_IminusJ[0][0] = np.sum(np.diagonal(p_ij))
    for i in range(1, p_ij.shape[0]):
        p_IminusJ[0][i] = np.sum(np.diag(p_ij, i)) * 2

    mu = get_DifferenceAvergae(GLCMmatrix)
    k = np.array(range(0, p_ij.shape[0]))
    return np.sum(((k - mu) ** 2) * p_IminusJ)


def get_DifferenceEntropy(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Diagonal Probability p_IminusJ
    p_IminusJ = np.zeros((1, p_ij.shape[0]))
    p_IminusJ[0][0] = np.sum(np.diagonal(p_ij))
    for i in range(1, p_ij.shape[0]):
        p_IminusJ[0][i] = np.sum(np.diag(p_ij, i)) * 2

    return -1 * np.sum(p_IminusJ * np.log2(p_IminusJ + 1e-7))


def get_SumAverage(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Cross-Diagonal Probability p_IplusJ
    p_ij90 = np.rot90(p_ij)
    p_IplusJ = np.zeros((1, 2 * p_ij.shape[0] - 1))
    for i in range(-(p_ij.shape[0]-1), p_ij.shape[0]):
        p_IplusJ[0][i + p_ij.shape[0] - 1] = np.sum(np.diagonal(p_ij90, i))

    k = np.array(range(2, 2 * p_ij.shape[0] + 1))
    # By definition, sum average = 2 * joint average
    # return 2 * get_JointAverage(GLCMmatrix)
    return np.sum(k * p_IplusJ)


def get_SumVaiance(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Cross-Diagonal Probability p_IplusJ
    p_ij90 = np.rot90(p_ij)
    p_IplusJ = np.zeros((1, 2 * p_ij.shape[0] - 1))
    for i in range(-(p_ij.shape[0] - 1), p_ij.shape[0]):
        p_IplusJ[0][i + p_ij.shape[0] - 1] = np.sum(np.diagonal(p_ij90, i))

    k = np.array(range(2, 2 * p_ij.shape[0] + 1))
    mu = get_SumAverage(GLCMmatrix)
    return np.sum(((k - mu) ** 2) * p_IplusJ)


def get_SumEntropy(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Cross-Diagonal Probability p_IplusJ
    p_ij90 = np.rot90(p_ij)
    p_IplusJ = np.zeros((1, 2 * p_ij.shape[0] - 1))
    for i in range(-(p_ij.shape[0] - 1), p_ij.shape[0]):
        p_IplusJ[0][i + p_ij.shape[0] - 1] = np.sum(np.diagonal(p_ij90, i))

    return -1 * np.sum(p_IplusJ * np.log2(p_IplusJ + 1e-7)) # plus a small number to avoid -Inf


def get_AngularSecondMomnet(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    return np.sum(p_ij ** 2)


def get_Contrast(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1]+1), range(1, p_ij.shape[0]+1))
    return np.sum(((row - col) ** 2) * p_ij)


def get_Dissimilarity(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    return np.sum(np.abs(row - col) * p_ij)


def get_InverseDifference(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    return np.sum(p_ij / (1 + np.abs(row - col)))


def get_NormalisedInverseDifference(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    Ng = p_ij.shape[0]
    return np.sum(p_ij / (1 + (np.abs(row - col) / Ng)))


def get_InverseDifferenceMoment(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    return np.sum(p_ij / (1 + (row - col) ** 2))


def get_NormalisedInverseDifferenceMoment(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    Ng = p_ij.shape[0]
    return np.sum(p_ij / (1 + ((row - col) ** 2) / (Ng ** 2)))


def get_InverseVarianve(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Diagonal Probability p_IminusJ
    p_IminusJ = np.zeros((1, p_ij.shape[0]))
    p_IminusJ[0][0] = np.sum(np.diagonal(p_ij))
    for i in range(1, p_ij.shape[0]):
        p_IminusJ[0][i] = np.sum(np.diag(p_ij, i)) * 2

    k = np.array(range(1, p_ij.shape[0]))
    p_IminusJ2 = np.delete(p_IminusJ, 0)
    return np.sum(p_IminusJ2 / (k ** 2))


def get_Correlation(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Marginal Probabilities
    p_i = np.sum(p_ij, 1)
    p_j = np.sum(p_ij, 0)
    # Mean of the marginal probability
    i = np.array(range(1, p_ij.shape[0]+1))
    j = np.array(range(1, p_ij.shape[1]+1))
    mu_i = np.sum(i * p_i)
    mu_j = np.sum(j * p_j)
    # Stamdard deviation of the marginal probability
    std_i = (np.sum(((i - mu_i) ** 2) * p_i)) ** (1/2)
    std_j = (np.sum(((j - mu_j) ** 2) * p_j)) ** (1/2)

    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))

    return np.sum((row - mu_i) * (col - mu_j) * p_ij) / (std_i * std_j)


def get_Autocorrelation(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    return np.sum(row * col * p_ij)

def get_ClusterTendency(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    # Marginal Probability
    p_i = np.sum(p_ij, 1)
    # Mean of the marginal probability
    i = np.array(range(1, p_ij.shape[0] + 1))
    mu_i = np.sum(i * p_i)

    return np.sum(((row + col - 2 * mu_i) ** 2) * p_ij)


def get_ClusterShade(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    # Marginal Probability
    p_i = np.sum(p_ij, 1)
    # Mean of the marginal probability
    i = np.array(range(1, p_ij.shape[0] + 1))
    mu_i = np.sum(i * p_i)

    return np.sum(((row + col - 2 * mu_i) ** 3) * p_ij)


def get_ClusterProminence(GLCMmatrix):
    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    col, row = np.meshgrid(range(1, p_ij.shape[1] + 1), range(1, p_ij.shape[0] + 1))
    # Marginal Probability
    p_i = np.sum(p_ij, 1)
    # Mean of the marginal probability
    i = np.array(range(1, p_ij.shape[0] + 1))
    mu_i = np.sum(i * p_i)

    return np.sum(((row + col - 2 * mu_i) ** 4) * p_ij)


def get_InformationCorrelation1(GLCMmatrix):
    # Joint Entropy
    HXY = get_JointEntropy(GLCMmatrix)

    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Marginal Probabilities
    p_i = np.sum(p_ij, 1)
    p_j = np.sum(p_ij, 0)

    HX = -1 * np.sum(p_i * np.log2(p_i + 1e-7)) # plus a small number to avoid -Inf
    p_i_flatten = np.repeat(p_i, p_ij.shape[0])
    p_j_flatten = np.tile(p_j, p_ij.shape[0])
    HXY1 = -1 * np.sum(p_ij.flatten() * np.log2(p_i_flatten * p_j_flatten + 1e-7)) # plus a small number to avoid -Inf

    return (HXY - HXY1) / HX


def get_InformationCorrelation2(GLCMmatrix):
    # Joint Entropy
    HXY = get_JointEntropy(GLCMmatrix)

    p_ij = GLCMmatrix / np.sum(GLCMmatrix)
    # Marginal Probabilities
    p_i = np.sum(p_ij, 1)
    p_j = np.sum(p_ij, 0)
    p_i_flatten = np.repeat(p_i, p_ij.shape[0])
    p_j_flatten = np.tile(p_j, p_ij.shape[0])
    HXY2 = -1 * np.sum(p_i_flatten * p_j_flatten * np.log2(p_i_flatten * p_j_flatten + 1e-7))

    return (1 - np.exp(-2 * (HXY2 - HXY))) ** (1/2)


def get_GLCMfeatures(ima):
    '''
    In this function, we calculate GLCM for 0 degree, 45 degree,
    90 degree and 135 degree with step = 1
    '''
    GLCM0 = get_GLCMmatrix(ima, 1, 0)
    GLCM45 = get_GLCMmatrix(ima, 1, 1)
    GLCM90 = get_GLCMmatrix(ima, 0, 1)
    GLCM135 = get_GLCMmatrix(ima, -1, 1)

    GLCM = GLCM0 + GLCM45 + GLCM90 + GLCM135

    Joint_Maximun = get_JointMaximum(GLCM)
    Joint_Average = get_JointAverage(GLCM)
    Joint_Varinace = get_JointVariance(GLCM)
    Joint_Entropy = get_JointEntropy(GLCM)
    Difference_Average = get_DifferenceAvergae(GLCM)
    Difference_Variance = get_DifferenceVariance(GLCM)
    Difference_Entropy = get_DifferenceEntropy(GLCM)
    Sum_Average = get_SumAverage(GLCM)
    Sum_Variance = get_SumVaiance(GLCM)
    Sum_Entrpy = get_SumEntropy(GLCM)
    Angular_Second_Moment = get_AngularSecondMomnet(GLCM)
    Contrast = get_Contrast(GLCM)
    Dissimilarity = get_Dissimilarity(GLCM)
    Inverse_Difference = get_InverseDifference(GLCM)
    Normalised_Inverse_Difference = get_NormalisedInverseDifference(GLCM)
    Inverse_Difference_Moment = get_InverseDifferenceMoment(GLCM)
    Normalised_Inverse_Difference_Moment = get_NormalisedInverseDifferenceMoment(GLCM)
    Inverse_Variance = get_InverseVarianve(GLCM)
    Correlation = get_Correlation(GLCM)
    Autocorrelation = get_Autocorrelation(GLCM)
    Cluster_Tendency = get_ClusterTendency(GLCM)
    Cluster_Shade = get_ClusterShade(GLCM)
    Cluster_Prominence = get_ClusterProminence(GLCM)
    Information_Correlation1 = get_InformationCorrelation1(GLCM)
    Information_Correlation2 = get_InformationCorrelation2(GLCM)

    return Joint_Maximun, Joint_Average, Joint_Varinace, Joint_Entropy,\
           Difference_Average, Difference_Variance, Difference_Entropy,\
           Sum_Average, Sum_Variance, Sum_Entrpy, Angular_Second_Moment,\
           Contrast, Dissimilarity, Inverse_Difference,\
           Normalised_Inverse_Difference, Inverse_Difference_Moment,\
           Normalised_Inverse_Difference_Moment, Inverse_Variance,\
           Correlation, Autocorrelation, Cluster_Tendency, Cluster_Shade,\
           Cluster_Prominence, Information_Correlation1, Information_Correlation2



























