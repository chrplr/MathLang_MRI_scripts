#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:58:50 2024

@author: mp271783
"""

import os.path as op
import glob
import pandas as pd
import numpy as np
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
import nibabel as nib

SENTENCE_MAPS = "../results/stat_maps/spm"
#SENTENCE_MAPS = "../results/contrasts_individual_maps_sentence_model"
SAVEDIR = "../results/group_analysis_contrasts_sentence_model"
SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
sentences = list(range(1,265))+list(range(266,321))

math_sentences = list(range(1,81))+list(range(241,265))+list(range(266,281))
language_sentences = list(range(121,161))+list(range(201,241))+list(range(281,321))
control_sentences = list(range(81,121))+list(range(161,201))

list_word_sentences = list(range(161,201))
colorlessg_sentences = list(range(81,121))
general_sentences = list(range(201,241))
contextual_sentences = list(range(121,161))
social_sentences = list(range(281,321))
calculus_sentences = list(range(1,41))
arithmetic_sentences = list(range(41,81))
geometry_sentences = list(range(241,265))+list(range(266,281))

dict_contrast = {'mathVSlang':[math_sentences, language_sentences],
                 'meaninglessVSlist_word':[colorlessg_sentences,list_word_sentences],
                 'generalVSmeaningless':[general_sentences, colorlessg_sentences],
                 'contextualVSgeneral':[contextual_sentences, general_sentences],
                 'socialVScontextual':[social_sentences, contextual_sentences],
                 'calculusVSothermath':[calculus_sentences, arithmetic_sentences+geometry_sentences],
                 'arithmeticVSothermath':[arithmetic_sentences, calculus_sentences+geometry_sentences],
                 'geometryVSothermath':[geometry_sentences, calculus_sentences+arithmetic_sentences]}

smoothing = 8.0

for num_sentence in sentences:
    print(f'computing sentence {num_sentence}')
    sentences_files = []
    for subj in SUBJECTS:
          sentences_files = sentences_files + glob.glob(op.join(SENTENCE_MAPS, f"sub{subj:02d}_sentence_{num_sentence}_*sentence_target.nii")) 

    n_subj = len(sentences_files)
    design_matrix_sentence = pd.DataFrame([1] * n_subj, columns=['intercept'])
#    print(nib.load(sentences_files[0]).shape)
    
    ############################################################################
      # Model specification and fit.
    second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
    second_level_model = second_level_model.fit(sentences_files, design_matrix=design_matrix_sentence)
    sentence_statmap= second_level_model.compute_contrast(np.array([1]),output_type="z_score")
    mapfile = op.join(SAVEDIR,f"sentence_{num_sentence}_smooth-{smoothing}_31_participants.nii")
    nib.save(sentence_statmap, mapfile)
    plot_stat_map(sentence_statmap, threshold=3.1, cut_coords=(-50, 10, 0), output_file = op.join(SAVEDIR,f'sentence_{num_sentence}_smooth-{smoothing}_31_participants.png'))


# for contrast in dict_contrast.keys():
#     print('computing contrast: ', contrast)
#     contrast_files = []
#     for subj in SUBJECTS:
#         contrast_files = contrast_files + glob.glob(op.join(SENTENCE_MAPS, f"sub-{subj:02d}_{contrast}_smooth-0.nii"))

#     n_subj = len(contrast_files)
#     design_matrix_sentence = pd.DataFrame([1] * n_subj, columns=['intercept'])
#     print(nib.load(contrast_files[0]).shape)
    
#     ############################################################################
#       # Model specification and fit.
#     second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
#     second_level_model = second_level_model.fit(contrast_files, design_matrix=design_matrix_sentence)
#     sentence_statmap= second_level_model.compute_contrast(np.array([1]),output_type="z_score")
#     mapfile = op.join(SAVEDIR,f"{contrast}_contrast_smooth{smoothing}_adults.nii")
#     nib.save(sentence_statmap, mapfile)

# for contrast in dict_contrast.keys():
#     print('computing contrast: ', contrast)
#     smoothing = 8.0
#     second_level_input = [op.join(f'../results/contrasts_individual_maps_sentence_model/sub-{subj:02d}_{contrast}_smooth-0.nii') for subj in SUBJECTS]
#     list_adultsVSados = [-1]*14 + [1]*15
#     design_matrix = pd.DataFrame(list_adultsVSados, columns = ['intercept'])
         
#         ############################################################################
#       # Model specification and fit.
#     second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
#     second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)
#     statmap = second_level_model.compute_contrast(output_type="z_score")
#     alpha = 0.05
#     map, threshold = threshold_stats_img(stat_img = statmap, alpha =alpha, height_control = 'fdr', cluster_threshold = 0, two_sided = True)
#     mapfile = op.join(SAVEDIR,f"adultsVSados_{contrast}_smooth-{smoothing}.nii")
#     nib.save(statmap, mapfile)  
