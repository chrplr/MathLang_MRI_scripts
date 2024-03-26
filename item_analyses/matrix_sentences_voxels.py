#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:48:29 2024

@author: mp271783

Create a matrix with all the sentences (x-axis) in one participant 
sorted by run by all the voxels in this participant (y-axis)
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os.path as op
import glob
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_multi_epi_mask, intersect_masks
import matplotlib.pyplot as plt
from nilearn.plotting import plot_epi, plot_roi, show, plot_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
SOURCEDIR = "../../Subjects_bids/fmriprep"
MASKDIR = f'{SOURCEDIR}/derivatives/fmri_prep/fmriprep'
SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
SENTENCE_MAPS = "../results/stat_maps/spm"
runs = [1,2,3,4,5]
sentences = list(range(1,265))+list(range(266,321))
math_sentences = list(range(1,81))+list(range(241,265))+list(range(266,281))
language_sentences = list(range(121,161))+list(range(201,241))+list(range(281,321))
control_sentences = list(range(81,121))+list(range(161,201))

masks_list =[]
for subj in SUBJECTS:
    print(f'Computing subject {subj} mask')
    for run in runs:
        file_id = f"sub-{subj:02d}_task-mathlang_run-{run}"
        if subj ==1 or subj ==2:
            mask_suffix = "space-MNI152NLin2009cAsym_desc-brain_mask_new_affine"
        else:
            mask_suffix = "space-MNI152NLin2009cAsym_desc-brain_mask"
        maskfile_run= op.join(MASKDIR,f"sub-{subj:02d}","func",f"{file_id}_{mask_suffix}.nii.gz")
        mask_load = nib.load(maskfile_run)
        mask_data =mask_load.get_fdata()
#        print(np.isnan(mask_data).any())
#        print(mask_load.shape)
        masks_list.append(maskfile_run)
#        plot_img(maskfile_run)
maskfile = intersect_masks(masks_list, threshold = 1)
print(maskfile.shape)
#plot_epi(maskfile, title="Mean EPI image")
mapfile = op.join("mask_31_participants_intersection.nii")
nib.save(maskfile, mapfile)
    
# for subj in SUBJECTS:
#     print(f'Computing total subject {subj} matrix')
#     n_voxels = int(np.sum(maskfile.get_fdata()))
#     matrix_sentencesXvoxels = np.empty((0,n_voxels))
#     for run in runs:
#         print(f'Computing run {run}')
#         sentence_files = glob.glob(op.join(SENTENCE_MAPS, f"sub{subj:02d}_sentence_*_run-{run}_sentence_target.nii"))
#         for file in sentence_files:
#             sentence_file = nib.load(file)
#             sentence_data = sentence_file.get_fdata()
#             masker = NiftiMasker(mask_img = maskfile)
#             sentence_transformed = masker.fit_transform(sentence_file)
#             matrix_sentencesXvoxels=np.vstack((matrix_sentencesXvoxels, sentence_transformed))
#     print(matrix_sentencesXvoxels.shape)
#     df = pd.DataFrame(data=matrix_sentencesXvoxels)
#     df.to_csv(f'sub-{subj}_matrix_sentencesXvoxels.csv', index=False)
# smoothing = 8.0
    
# n_voxels = int(np.sum(maskfile.get_fdata()))
# matrix_sentencesXvoxels = np.empty((0,n_voxels))
# for num_sentence in language_sentences:
#     print(f'computing sentence {num_sentence}')
#     sentences_files = []
#     for subj in SUBJECTS:
#         sentences_files = sentences_files + glob.glob(op.join(SENTENCE_MAPS, f"sub{subj:02d}_sentence_{num_sentence}_*sentence_target.nii")) 
#     n_subj = len(sentences_files)
#     design_matrix_sentence = pd.DataFrame([1] * n_subj, columns=['intercept'])
#     print(nib.load(sentences_files[0]).shape)
    
#     ############################################################################
#       # Model specification and fit.
#     second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
#     second_level_model = second_level_model.fit(sentences_files, design_matrix=design_matrix_sentence)
#     sentence_statmap= second_level_model.compute_contrast(np.array([1]),output_type="z_score")
# #    matrix_4D_sentencesXvoxels=np.vstack((matrix_sentencesXvoxels, sentence_statmap))
# #    sentence_file = nib.load(sentence_statmap)
# #    sentence_data = sentence_file.get_fdata()
#     masker = NiftiMasker(mask_img = maskfile)
#     sentence_transformed = masker.fit_transform(sentence_statmap)
#     matrix_sentencesXvoxels=np.vstack((matrix_sentencesXvoxels, sentence_transformed))
# print(matrix_sentencesXvoxels.shape)
# df = pd.DataFrame(data=matrix_sentencesXvoxels)
# df.to_csv('sub-total_language_matrix_sentencesXvoxels.csv', index=False) 
    #plot_stat_map(sentence_statmap, threshold=3.1, cut_coords=(-50, 10, 0), output_file = f'sentence_{num_sentence}_smooth{smoothing}.png')
    