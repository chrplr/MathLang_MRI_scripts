#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:41:13 2024

@author: mp271783
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os.path as op
import glob
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_multi_epi_mask
import matplotlib.pyplot as plt
from nilearn.plotting import plot_epi, plot_roi, show, plot_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
from npica import ICA
from sklearn.decomposition import PCA

SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
runs = [1,2,3,4,5]
SOURCEDIR = "../../Subjects_bids/fmriprep"
MASKDIR = '../results/Masks_ROIs_Harvard_Oxford'
SENTENCE_MAPS = '../results/Group_analysis_individual_sentence'
components =list(range(0,9))

def reverse_transform(matrix_compXvoxels, maskfile):
    matrix_transformed = masker.inverse_transform(matrix_compXvoxels)
    return matrix_transformed
mask = "mask_31_participants_intersection.nii"
maskfile = nib.load(mask)
n_voxels = int(np.sum(maskfile.get_fdata()))
matrix_sentencesXvoxels=np.empty((0,n_voxels))
all_sentences = list(range(1,265))+list(range(266,321))
#glob.glob(op.join(SENTENCE_MAPS, "sentence_*_smooth-8.0_31_participants.nii"))
masker = NiftiMasker(mask_img = maskfile)

for sentence in all_sentences:
    print(f'computing_sentence-{sentence}')
    sentence_file = op.join(SENTENCE_MAPS, f"sentence_{sentence}_smooth-8.0_31_participants.nii")
    sentence_transformed = masker.fit_transform(sentence_file)
    matrix_sentencesXvoxels=np.vstack((matrix_sentencesXvoxels, sentence_transformed))
    df = pd.DataFrame(data=matrix_sentencesXvoxels)
    df.to_csv('sub-total_matrix_31_participants_sentencesXvoxels.csv', index=False)
print(matrix_sentencesXvoxels.shape)

# K=len(components)
# X = matrix_sentencesXvoxels.T
# N_RANDOM_INITS = 100000
# RAND_SEED = 0
# ica = ICA(K=K,N_RANDOM_INITS= N_RANDOM_INITS, RAND_SEED=RAND_SEED)
# ica.fit(X)
# npica_data = ica.sources.T

ncomponents = len(components)
pca = PCA(n_components = ncomponents)
data_pca = pca.fit_transform(matrix_sentencesXvoxels)

matrix_pca_componentsXvoxels = np.dot(data_pca.T,matrix_sentencesXvoxels)
# Normalize estimated components, for thresholding to make sense
matrix_pca_componentsXvoxels-=matrix_pca_componentsXvoxels.mean(axis=0)
matrix_pca_componentsXvoxels/=matrix_pca_componentsXvoxels.std(axis=0)
print(matrix_pca_componentsXvoxels.shape)

for comp in components:
    matrix_comp = np.reshape(matrix_pca_componentsXvoxels[comp,],(1,n_voxels))
    matrix_roi = reverse_transform(matrix_comp,maskfile)
    print(matrix_roi.shape)
    plot_stat_map(matrix_roi, threshold=0.0, output_file = f'component-{comp+1}_PCA_in_31_participants.png')
    mapfile = op.join(f"comp-{comp+1}_PCA_in_31_participants.nii")
    nib.save(matrix_roi, mapfile)
    