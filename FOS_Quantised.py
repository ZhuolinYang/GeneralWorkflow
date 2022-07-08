# Intensity Histogram Features
'''
Following the IBSI, there are 23 features in this category.
Intensity histogran features require image quantisation (default 16).
'''

import numpy as np

def Quantisation(ima, quantise = 16):
    min = np.nanmin(ima)
    max = np.nanmax(ima)
    ima_quantised = np.trunc((ima-min) / ((max - min) / quantise)) + 1
    ima_quantised[np.where(ima_quantised == np.max(ima_quantised))] = quantise
    return ima_quantised


def get_hist(ima):
    Nbins = np.max(ima) - np.min(ima) + 1
    hist,_ = np.histogram(ima.flatten(), bins = int(Nbins))
    p_i = hist / ima.size # occurrence probability
    return hist, p_i


def get_Mean_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanmean(ima_quantised)


def get_Variance_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanvar(ima_quantised)


def get_Skewness_quantised(ima):
    ima_quantised = Quantisation(ima)
    var = get_Variance_quantised(ima_quantised)
    if var == 0:
        Skewness = 0
    else:
        n = ima.size
        mu = np.nanmean(ima_quantised)
        num = np.sum((ima_quantised - mu) ** 3) / n
        den = (np.sum((ima_quantised - mu) ** 2) / n) ** (3 / 2)
        Skewness = num / den
    return Skewness


def get_Kurtosis_quantised(ima):
    ima_quantised = Quantisation(ima)
    var = get_Variance_quantised(ima_quantised)
    if var == 0:
        Kurtosis = 0
    else:
        n = ima.size
        mu = np.nanmean(ima_quantised)
        num = np.sum((ima_quantised - mu) ** 4) / n
        den = (np.sum((ima_quantised - mu) ** 2) / n) ** 2
        Kurtosis = (num / den) - 3
    return Kurtosis


def get_Median_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanmedian(ima_quantised)


def get_Minimun_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanmin(ima_quantised)


def get_10Percentile_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanpercentile(ima_quantised,10)


def get_90Percentile_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanpercentile(ima_quantised,90)


def get_Maximun_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanmax(ima_quantised)


def get_IntensityHistogramMode_quantised(ima):
    ima_quantised = Quantisation(ima)
    hist, _ = get_hist(ima_quantised)
    mu = np.nanmean(ima_quantised)
    # find the most common discretised intensities
    max_index = []
    max_value = np.max(hist)
    for i in range(len(hist)):
        if hist[i] == max_value:
            max_index.append(i)
    most_common_intensities = [(x+1) for x in max_index]
    # if there are multiple most common intensities, choose the one that is closest to the mean
    if len(most_common_intensities) == 1:
        Mode = most_common_intensities[0]
    elif len(most_common_intensities) > 1:
        distance_to_mean = [np.abs(x - mu) for x in most_common_intensities]
        index = []
        min_distance = min(distance_to_mean)
        for i in range(len(distance_to_mean)):
            if distance_to_mean[i] == min_distance:
                index.append(i)
    # if there are two intensities equidistant to the mean, the one to the left to the mean is chosen
            if len(index) == 1:
                Mode = most_common_intensities[index[0]]
            elif len(index) > 1:
                min_index = min(index)
                Mode = most_common_intensities[min_index]

    return Mode


def get_InterquartileRange_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanpercentile(ima_quantised,75) - np.nanpercentile(ima_quantised,25)


def get_IntensityRange_quantised(ima):
    ima_quantised = Quantisation(ima)
    return np.nanmax(ima_quantised) - np.nanmin(ima_quantised)


def get_MeanAbsoluteDeviation_quantised(ima):
    ima_quantised = Quantisation(ima)
    n = ima_quantised.size
    mu = np.nanmean(ima_quantised)
    return np.sum(np.abs(ima_quantised - mu)) / n


def get_RobustMeanAbsoluteDeviation_quantised(ima):
    ima_quantised = Quantisation(ima)
    percent10 = np.nanpercentile(ima_quantised, 10)
    percent90 = np.nanpercentile(ima_quantised, 90)
    percentArray = [x for x in ima_quantised.flatten() if ((x-percent10 >= 0) and (x-percent90 <= 0))]
    percentArray = np.array(percentArray)
    n = len(percentArray)
    mu = np.nanmean(percentArray)
    return np.sum(np.abs(percentArray - mu)) / n


def get_MedianAbsoluteDeviation_quantised(ima):
    ima_quantised = Quantisation(ima)
    n = ima_quantised.size
    m = np.nanmedian(ima_quantised)
    return np.sum(np.abs(ima_quantised - m)) / n


def get_CoefficientOfVariation_quantised(ima):
    ima_quantised = Quantisation(ima)
    sigma = np.nanvar(ima_quantised) ** (1/2)
    mu = np.nanmean(ima_quantised)
    return sigma / mu


def get_QuartileCoefficientOfDispersion_quantised(ima):
    ima_quantised = Quantisation(ima)
    percent75 = np.nanpercentile(ima_quantised,75)
    percent25 = np.nanpercentile(ima_quantised,25)
    return (percent75 - percent25) / (percent75 + percent25)


def get_Entropy_quantised(ima):
    ima_quantised = Quantisation(ima)
    _, p_i = get_hist(ima_quantised)
    return -1 * np.sum(p_i * np.log2(p_i + 1e-7)) # plus a small number to avoid -Inf


def get_Uniformity_quantised(ima):
    ima_quantised = Quantisation(ima)
    _, p_i = get_hist(ima_quantised)
    return np.sum(p_i ** 2)


def get_MaximunHistogramGradient_quantised(ima):
    ima_quantised = Quantisation(ima)
    hist,_ = get_hist(ima_quantised)
    hist_grad = np.zeros(len(hist))
    hist_grad[0] = hist[1] - hist[0]
    Ng = len(hist_grad)
    for i in range(1, Ng-1):
        hist_grad[i] = (hist[i+1] - hist[i-1]) / 2
    hist_grad[-1] = hist[-1] - hist[-2]
    Maximum_Gradient = np.max(hist_grad)

    return Maximum_Gradient


def get_MaximumHistogramGradientIntensity_quantised(ima):
    ima_quantised = Quantisation(ima)
    hist,_ = get_hist(ima_quantised)
    hist_grad = np.zeros(len(hist))
    hist_grad[0] = hist[1] - hist[0]
    Ng = len(hist_grad)
    for i in range(1, Ng-1):
        hist_grad[i] = (hist[i+1] - hist[i-1]) / 2
    hist_grad[-1] = hist[-1] - hist[-2]
    Maximum_Gradient_Intensity = np.where(hist_grad == np.max(hist_grad))[0][0] + 1

    return Maximum_Gradient_Intensity


def get_MinimunHistogramGradient_quantised(ima):
    ima_quantised = Quantisation(ima)
    hist,_ = get_hist(ima_quantised)
    hist_grad = np.zeros(len(hist))
    hist_grad[0] = hist[1] - hist[0]
    Ng = len(hist_grad)
    for i in range(1, Ng-1):
        hist_grad[i] = (hist[i+1] - hist[i-1]) / 2
    hist_grad[-1] = hist[-1] - hist[-2]
    Minimum_Gradient = np.min(hist_grad)

    return Minimum_Gradient


def get_MinimumHistogramGradientIntensity_quantised(ima):
    ima_quantised = Quantisation(ima)
    hist,_ = get_hist(ima_quantised)
    hist_grad = np.zeros(len(hist))
    hist_grad[0] = hist[1] - hist[0]
    Ng = len(hist_grad)
    for i in range(1, Ng-1):
        hist_grad[i] = (hist[i+1] - hist[i-1]) / 2
    hist_grad[-1] = hist[-1] - hist[-2]
    Minimum_Gradient_Intensity = np.where(hist_grad == np.min(hist_grad))[0][0] + 1

    return Minimum_Gradient_Intensity


def get_FOSfeatures_quantised(ima):
    Mean = get_Mean_quantised(ima)
    Variance = get_Variance_quantised(ima)
    Skewness = get_Skewness_quantised(ima)
    Kurtosis = get_Kurtosis_quantised(ima)
    Median = get_Median_quantised(ima)
    Minimum = get_Minimun_quantised(ima)
    Percentile10 = get_10Percentile_quantised(ima)
    Percentile90 = get_90Percentile_quantised(ima)
    Maximum = get_Maximun_quantised(ima)
    Mode = get_IntensityHistogramMode_quantised(ima)
    Interquartile_Range = get_InterquartileRange_quantised(ima)
    Intensity_Range = get_IntensityRange_quantised(ima)
    Mean_Absolute_deviation = get_MeanAbsoluteDeviation_quantised(ima)
    Robust_Mean_Absolute_Deviation = get_RobustMeanAbsoluteDeviation_quantised(ima)
    Median_Absolute_Deviation = get_MedianAbsoluteDeviation_quantised(ima)
    Coefficient_Of_Variation = get_CoefficientOfVariation_quantised(ima)
    Quartile_Coefficient_Of_Dispersion = get_QuartileCoefficientOfDispersion_quantised(ima)
    Entropy = get_Entropy_quantised(ima)
    Uniformity = get_Uniformity_quantised(ima)
    Maximum_Histogram_Gradient = get_MaximunHistogramGradient_quantised(ima)
    Maximum_Histogram_Gradient_Intensity = get_MaximumHistogramGradientIntensity_quantised(ima)
    Minimum_Histogram_Gradient = get_MinimunHistogramGradient_quantised(ima)
    Minimum_Histogram_Gradient_Intensity = get_MinimumHistogramGradientIntensity_quantised(ima)

    return Mean, Variance, Skewness, Kurtosis, Median, Minimum,\
           Percentile10, Percentile90, Maximum, Mode, Interquartile_Range,\
           Intensity_Range, Mean_Absolute_deviation, Robust_Mean_Absolute_Deviation,\
           Median_Absolute_Deviation, Coefficient_Of_Variation, Quartile_Coefficient_Of_Dispersion,\
           Entropy, Uniformity, Maximum_Histogram_Gradient, Maximum_Histogram_Gradient_Intensity,\
           Minimum_Histogram_Gradient, Minimum_Histogram_Gradient_Intensity












