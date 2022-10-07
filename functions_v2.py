import os
import re
import pydicom as dicom
import dicom_contour.contour as dcm
import math
import numpy as np
from scipy.sparse import csc_matrix
import scipy.ndimage as scn
from copy import deepcopy
import warnings
import cv2
import openpyxl
import FOS_Quantised
import GLCM
import GLRLM
import GLSZM
import GLDZM
import NGTDM
import NGLDM
import scipy.io as scio
from scipy.interpolate import interpn


# Get the contour file from the current path
def get_contour_file(path):
    '''
    Get contour file from the current path

    Input:
    path (str) - the path that contains all the DICOM files

    Return:
    contour_file (str) - the path of the contour file

    '''
    DicomFiles = []

    for (root, dirs, files) in os.walk(path):
        for filename in files:
            if ".dcm" in filename:
                DicomFiles.append(os.path.join(root, filename))

    n = 0
    for FileNames in DicomFiles:
        file = dicom.read_file(FileNames)
        if 'ROIContourSequence' in dir(file):
            contour_file = FileNames
            n = n + 1

    if n > 1:
        warnings.warn("There are more than one contour files, returning the last one!")

    return contour_file


# Get the right contour sequence
def get_contour_sequence(contour_file, contour_name):
    '''
    The contour file will embed information in a
    unstructured tree-like manner. After reading
    this Dicom file using pydicom.read_file(), we
    can see the nested structure. The contour file
    may contain different sequences, e.g. various
    contours from differnet experts.

    Input:
    contour_file (str) - the path of the contour file
                         is the result from function
                         get_contour_file
    contour_name (str) - the name of the contour that
                         you are looking for, for example,
                         'rectum', 'bladder'

    Return:
    ROIContourSeq (int) - the index of the specified contour in
                          the contour seqence
    '''
    target_contour = []
    contour_data = dicom.read_file(contour_file)
    contour_sequence = dcm.get_roi_names(contour_data)
    contour_sequence = [s.lower() for s in contour_sequence]
    for contour in contour_sequence:
        if contour_name in contour:
            # Here you will need to set some rules to find the
            # correct contours, for example, sometimes ‘rectum’
            # and rectum_avoid’ will exist in a contour sequence
            # in the same time, so we have to set a rule to exclude\
            # ‘rectum_avoid’
            if not "avoid" in contour:
                target_contour.append(contour)

    ROIContourSeq = contour_sequence.index(target_contour[0])

    return ROIContourSeq


# Get the right order of the slices of the DICOM images
def order_slice(path):
    '''
    Return a list taht contains ordered image

    Input:
    path (str) - path that contains Dicom images

    Return:
    ordered_slices (dict) - ordered tuples of filename, index and z-position
    '''
    ImageFiles = []

    for (root, dirs, files) in os.walk(path):
        for filename in files:
            if ".dcm" in filename.lower(): # specify that we want to find dcm files
                if not "RS" in filename:
                    ImageFiles.append(filename)

    # sort the images files based on their z positions
    # read in each image iteratively
    slices = [dicom.read_file(path + '/' + f) for f in ImageFiles]
    # create a dictionaty to store all images and their corresponding z positions
    slice_dict = {f: dicom.read_file(path + '/' + f).ImagePositionPatient[2] for f in ImageFiles}
    ordered = sorted(slice_dict.items(), key=lambda x: x[1], reverse=True)
    ordered_slices = [(a, b, c) for a, b, c in
                      zip(np.array(ordered)[:, 0], range(len(ordered)), np.array(ordered)[:, 1])]
    ordered_slices = np.array(ordered_slices)
    ordered_slices

    return ordered_slices


def cartesian2pixels(contour_dataset, path):
    '''
        Return image pixel array and contour label array given a contour dataset and the path
        that contains the image files

        Inputs:
        contour_dataset - DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path (str) - the path that contains all the Dicom files

        Return:
        ima_array - 2d numpy array of image with pixel intensities
        contour_array - 2d numpy array of contour with labels 0 and 1
        '''

    contour_coord = contour_dataset.ContourData

    # x, y, z coordinates of the contour in mm
    x0 = contour_coord[len(contour_coord) - 3]
    y0 = contour_coord[len(contour_coord) - 2]
    z0 = contour_coord[len(contour_coord) - 1]
    coord = []
    for i in range(0, len(contour_coord), 3):
        x = contour_coord[i]
        y = contour_coord[i + 1]
        z = contour_coord[i + 2]
        l = math.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
        l = math.ceil(l * 2) + 1  # ceil: round toward positive infinity
        for j in range(1, l + 1):
            coord.append([(x - x0) * j / l + x0, (y - y0) * j / l + y0, (z - z0) * j / l + z0])
        x0 = x
        y0 = y
        z0 = z

    # Extract the image id corresponding to given contour
    ima_file = []
    for (root, dirs, files) in os.walk(path):
        for filename in files:
            if ".dcm" in filename:
                if not "RTDOSE" in filename:
                    if not "RTSTRUCT" in filename:
                        if not "RTPLAN" in filename:
                            ima_file.append(filename)

    correspond_ima_file = []
    for FileNames in ima_file:
        f = dicom.read_file(path + '/' + FileNames)
        if f.SOPInstanceUID == contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID:
            correspond_ima_file.append(FileNames)

    # Read that Dicom image file
    ima = dicom.read_file(path + '/' + correspond_ima_file[0])
    ima_array = ima.pixel_array

    # Physical distance between the center of each pixel
    x_spacing = float(ima.PixelSpacing[0])
    y_spacing = float(ima.PixelSpacing[1])

    # The centre of the the upper left voxel
    origin_x = ima.ImagePositionPatient[0]
    origin_y = ima.ImagePositionPatient[1]
    origin_z = ima.ImagePositionPatient[2]

    # mapping
    pixel_coords = [(np.round((y - origin_y) / y_spacing), np.round((x - origin_x) / x_spacing)) for x, y, _ in coord]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_array = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                               shape=(ima_array.shape[0], ima_array.shape[1])).toarray()

    return ima_array, contour_array, correspond_ima_file


def ContourImaArray(contour_file, path, ROIContourSeq):
    '''
    Return the arrays of the contour and the corresponding images given the contour file and
    the path of the images.

    Inputs:
    contour_file (str) - the path of the contour file
    path (str) - the path that contains all the Dicom files
    ROIContourSeq (int) - shows which sequence of contouring to use, default 5 (Rectum)

    Return:
    contour_ima_arrays (list) - a list that contains pairs of image pixel array and contour label array
    '''
    contour_data = dicom.read_file(contour_file)
    Rectum = contour_data.ROIContourSequence[ROIContourSeq]
    # get contour dataset in a list
    contours = [contour for contour in Rectum.ContourSequence]
    contour_ima_arrays = [cartesian2pixels(cdata, path) for cdata in contours]
    number_of_correspond_ima = len(contours)

    return contour_ima_arrays, number_of_correspond_ima


def get_contour_dict(contour_file, path, ROIContourSeq):
    '''
    Return a dictionary as key: image filename, value: [corresponding image array, corresponding contour array]

    Input:
    contour_file (str) - contour file name
    path (str) - path contains all the Dicom files
    ROIContourSeq (int) - shows which sequence of contouring to use, default 5 (Rectum)

    Return:
    contour_dict: dictionary with 2d numpy array
    '''
    contour_list, _ = ContourImaArray(contour_file, path, ROIContourSeq)

    contour_dict = {}
    for ima_arr, contour_arr, ima_id in contour_list:
        contour_dict[ima_id[0]] = [ima_arr, contour_arr]

    return contour_dict


def get_slices_with_contours(contour_file, path, ROIContourSeq, ordered_slices):
    '''
    Get the slice file names and their corresponding indices that contains the contour

    Input:
    contour_dict (dict) - the contour dictionary which is the
                          output of function get_contour_dict
    ordered_slices (ndarray) - filenames, indices and z positions of ordered slices which
                               are the output of function order_slice

    '''
    contour_dict = get_contour_dict(contour_file, path, ROIContourSeq)

    slicenames_with_contours = [k for k, v in contour_dict.items() if 1 in v[1]]

    slices_with_contours = []
    for i in range(len(slicenames_with_contours)):
        for j in range(len(ordered_slices)):
            if slicenames_with_contours[i] == ordered_slices[j][0]:
                slices_with_contours.append((ordered_slices[j][0], int(ordered_slices[j][1])))

    slices_with_contours.sort(key=lambda x: x[1]) # sort by indices

    return slices_with_contours


def contour_expansion(mask, width):
    '''
    Return the expanded contour with the width you set

    Inputs:
    mask (np.array) - the mask array with 0s and 1s, the mask is the solid contour.
    width (int) - the width you want your contour to be with

    Return:
    expanded_contour_array (np.array)
    '''
    m, n = mask.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    expanded_contour_array = deepcopy(mask.astype('uint8'))
    erosion = mask.astype('uint8')
    i = 0
    while i < math.ceil(max(m, n) / 2) - 1:
        erosion = cv2.erode(erosion, kernel, iterations=1)
        index = np.argwhere(erosion == 1)
        for j in range(len(index)):
            expanded_contour_array[int(index[j][0]), int(index[j][1])] += 1
        i += 1
    expanded_contour_array[(expanded_contour_array >= 1) & (expanded_contour_array <= width)] = 1
    expanded_contour_array[expanded_contour_array > width] = 0

    return expanded_contour_array


def get_image_expandedContour_files(path, ROISequence, width):
    '''
    Return a 3d array of images and corresponding 3d contours, masks and expanded cotours.

    Inputs:
    path (str) - path that contains all the Dicom files
    ROIContourSeq (int) - shows which sequence of contouring to use, default 5 (Rectum)
    width (int) - the width that you want your contour to be with.

    Return:
    images, contours, masks and expanded contours np.arrays
    '''

    # Extract arrays from DICOM

    images = []
    contours = []
    masks = []
    expandedContours = []

    # get contour file
    contour_file = get_contour_file(path)
    # get ordered slice
    ordered_slices = order_slice(path)
    # get contour dictionary
    contour_dict = get_contour_dict(contour_file, path, ROISequence)

    for key, value1, value2 in ordered_slices:
        #  get data from contour dict
        if key in contour_dict:
            images.append(contour_dict[key][0])
            contour = contour_dict[key][1]
            contours.append(contour)
            mask = (scn.binary_fill_holes(contour) if contour.max() == 1 else contour).astype(int)
            masks.append(mask)
            expandedContour = contour_expansion(mask, width)
            expandedContours.append(expandedContour)
        # get data from dicom.read_file
        else:
            ima_array = dicom.read_file(path + '/' + key).pixel_array
            contour_array = np.zeros_like(ima_array)
            mask_array = np.zeros_like(ima_array)
            expandedContour_array = np.zeros_like(ima_array)
            images.append(ima_array)
            contours.append(contour_array)
            masks.append(mask_array)
            expandedContours.append(expandedContour_array)

    return np.array(images), np.array(contours), np.array(masks), np.array(expandedContours)


def expand_two_sides(obj, outward_expansion_size, target_size):
    '''
    Aim: expand a ring-like object(2D) inwards and outwards to make it reach the target size.
         For example, expand a 2-pixel-wide organ wall inwards and outwards with 3 pixels to
         generate a 8-pixel-wide organ wall.
         In this case, the obj would be the organ mask (rather than the 2-pixel-wide contour,
         would explain this later), the outward_expansion_size would be 3, the target_size
         would be 8.

    How it works: the funtion first expand the whole mask (obj) outwards with 3 pixels (outward_
                  expandsion_size), then by calling the function 'contour_expansion', the
                  new expanded contour is generated, on this basis, the 8-pixel-wide organ wall
                  will be extracted by exapnding this new expanded contour inwards with 8 pixels
                  (target_size).
                  In this way, you can also control the proportion fo expansion on sides.
                  For example, you can set outward_expansion_size = 2, target_size = 8, then in
                  our case, the 8-pixel-wide organ wall will be generated by expanding the 2-pixel-
                  wide organ wall outwards with 2 pixels and inwards with 4 pixels.

    Input:
    obj: 2D array. In our case, should be the whole organ mask.
    outward_expansion_size: int
    target_size: int

    Output:
    expanded_obj: 2D array.

    '''
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_dilate = cv2.dilate(obj.astype('uint8'), kernel, outward_expansion_size)  # exapnd the mask outwards
    expanded_obj = contour_expansion(mask_dilate, target_size)  # expand the mask inwards
    return expanded_obj



def generate_features(ima, mask, ROI_size, ifHollow = True, outward_expansion_size = 3, positionRec = 'upper left'):
    '''
    calculate IBSI-defined radiomic features
    ima: 2d array, slice used to calculate features
    mask: 2d array, region of interest used to calculate features
    ROI_size: int, size of subimages
    positionRec: got 2 choices.
                 'upper left': record only the position of the upper left pixel of subimages, only applicable when the size
                               of all the subimages is regular and fixed, eg. 8*8, 16*16 (record only the position of the
                               upper left pixel because you've already known the width and length of the subimages)
                 'all': record the positions of all pixels in subimages, applicable to all cases, but only recomended to do this
                        when (1) the shapes of subimages are irregular; (2) when you want to have subsequent processing on the features,
                        for exmaple, you may want to average the features from different subimages within a region, in this situation, what you
                        want to record is the position of the region where the features are averaged, since you are using the averaged features to do the
                        analysis. Feature set of subimage1: F1 = {f11, f12, f13, ..., (14, 15, 28, 29)}, feature set of subimage2:
                        F2 = {f21, f22, f23, ..., (56, 57, 70)}, the array at the end would be the the position information for subimages.
                        Averaged features F = {f1, f2, f3, ..., (24, 25, 28, 29, 56, 57, 78)} --> What you nned to do is do combine the
                        position information array together, and don't forget to use np.unqie to get the unique elements of the array just
                        in case there are overlaps.

    '''

    features = []
    # for hollow organ, need to expand the rectal wall inwards and outwards to generate subimages
    if ifHollow is True:
        msk = expand_two_sides(mask, outward_expansion_size, ROI_size)
    # for solid organ, no need to do expansion
    else:
        msk = mask

    # create a pixel map to help markdown the position information of subimages (the region you used to calculate features)
    pixel_total = ima.shape[0] * ima.shape[1]
    pixel_arr = np.arange(pixel_total)
    pixel_map = pixel_arr.reshape((ima.shape[0], ima.shape[1]))

    ROI_mask = np.ones((ROI_size, ROI_size))
    # generate multiple n*n ROIs on the expanded rectal wall using convolution
    ima_h, ima_w = msk.shape
    kernel_h, kernel_w = ROI_mask.shape
    new_h = ima_h - kernel_h + 1
    new_w = ima_w - kernel_w + 1

    for i in range(new_h):
        for j in range(new_w):
            multiply = msk[i: i + kernel_h, j: j + kernel_w] * ROI_mask
            if np.sum(multiply) == ROI_size ** 2:
                # texture features
                ima_ROI = ima[i: i + kernel_h, j: j + kernel_w]
                pixel_ROI = pixel_map[i: i + kernel_h, j: j + kernel_w]
                # if (ima_ROI < 800).any() == False:  # remove ROIs that contaions gas using threshold 800
                if (ima_ROI == ima[i,j]).all() == False: # if the all elements in the ima_ROI are the same, skip that ima_ROI
                    FOS_features = FOS_Quantised.get_FOSfeatures_quantised(ima_ROI)
                    GLCM_features = GLCM.get_GLCMfeatures(ima_ROI)
                    GLRLM_features = GLRLM.get_GLRLMfeatures(ima_ROI)
                    GLSZM_features = GLSZM.get_GLSZMfeatures(ima_ROI)
                    GLDZM_features = GLDZM.get_GLDZMfeatures(ima_ROI)
                    NGTDM_features = NGTDM.get_NGTDMfeatures(ima_ROI)
                    NGLDM_features = NGLDM.get_NGLDMfeatures(ima_ROI)
                    if positionRec == 'upper left':
                        position = (pixel_ROI[0][0],)
                    elif positionRec == 'all':
                        position = tuple(pixel_ROI.flatten())

                    features.append([FOS_features, GLCM_features, GLRLM_features, GLSZM_features, GLDZM_features,
                                   NGTDM_features, NGLDM_features, position])

    return features

