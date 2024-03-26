#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:43:37 2023

@author: mp271783
"""

import os.path as op
import nibabel as nib
import glob
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain


SENTENCE_MAPS = "../results"
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

SAVEDIR = "../results/"
#SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]

dict_contrast = {'mathVSlang':[math_sentences, language_sentences],
                 'meaninglessVSlist_word':[colorlessg_sentences,list_word_sentences],
                 'generalVSmeaningless':[general_sentences, colorlessg_sentences],
                 'contextualVSgeneral':[contextual_sentences, general_sentences],
                 'socialVScontextual':[social_sentences, contextual_sentences],
                 'calculusVSothermath':[calculus_sentences, arithmetic_sentences+geometry_sentences],
                 'arithmeticVSothermath':[arithmetic_sentences, calculus_sentences+geometry_sentences],
                 'geometryVSothermath':[geometry_sentences, calculus_sentences+arithmetic_sentences]}

for contrast in dict_contrast.keys():
    print('computing contrast: ', contrast)
    smoothing = 3.0
    contrast_first_files = []
    for num_sentence in dict_contrast[contrast][0]:
        contrast_first_files = contrast_first_files + glob.glob(op.join(SENTENCE_MAPS, f"sentence-{num_sentence}_adultsVSados_smooth-8.0.nii")) 
    
    contrast_second_files = []
    for num_sentence in dict_contrast[contrast][1]:
        contrast_second_files = contrast_second_files + glob.glob(op.join(SENTENCE_MAPS, f"sentence-{num_sentence}_adultsVSados_smooth-8.0.nii")) 

    
    n_first_contrast = len(contrast_first_files)
    n_second_contrast = len(contrast_second_files)
    
    design_matrix = pd.DataFrame(dict(inter=[1] * (n_first_contrast + n_second_contrast), cont=[1]* n_first_contrast + [0]*n_second_contrast))
         
        ############################################################################
      # Model specification and fit.
    second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
    second_level_model = second_level_model.fit(contrast_first_files + contrast_second_files, design_matrix=design_matrix)
    statmap = second_level_model.compute_contrast(np.array([0,1]), output_type="z_score")
    mapfile = op.join(SAVEDIR,f"{contrast}_contrast_smooth-{smoothing}.nii")
    nib.save(statmap, mapfile)                                                  
    plot_stat_map(statmap, threshold=3.1121, cut_coords=(-50, 10, 0), output_file = f'../results/{contrast}_contrast_smooth-{smoothing}.png')
