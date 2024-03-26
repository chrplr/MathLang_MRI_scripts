#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:48:07 2024

@author: mp271783
"""

import os.path as op
import nibabel as nib
import glob
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
from nilearn.plotting import plot_stat_map


SENTENCE_MAPS = "../results/stat_maps/spm"
SAVEDIR = "../results/"
SUBJECTS_ADULTS = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]
SUBJECTS_ADOS=[101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]

sentences = list(range(1,265))+list(range(266,321))

for num_sentence in sentences:
    print('computing sentence: ', num_sentence)
    smoothing = 8.0
    adults_files = []
    for adults_subj in SUBJECTS_ADULTS:
        adults_files = adults_files + glob.glob(op.join(SENTENCE_MAPS, f"sub{adults_subj:02d}_sentence_{num_sentence}_*sentence_target.nii")) 
    
    ados_files = []
    for ados_subj in SUBJECTS_ADOS:
        ados_files = ados_files + glob.glob(op.join(SENTENCE_MAPS, f"sub{ados_subj:02d}_sentence_{num_sentence}_*sentence_target.nii")) 

    
    n_first_contrast = len(adults_files)
    n_second_contrast = len(ados_files)
    
    design_matrix = pd.DataFrame(dict(inter=[1] * (n_first_contrast + n_second_contrast), cont=[1]* n_first_contrast + [0]*n_second_contrast))
         
        ############################################################################
      # Model specification and fit.
    second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
    second_level_model = second_level_model.fit(adults_files + ados_files, design_matrix=design_matrix)
    statmap = second_level_model.compute_contrast(np.array([0,1]), output_type="z_score")
    mapfile = op.join(SAVEDIR,f"sentence-{num_sentence}_adultsVSados_smooth-{smoothing}.nii")
    nib.save(statmap, mapfile)                                                  
    plot_stat_map(statmap, threshold=3.1121, cut_coords=(-50, 10, 0), output_file = f'sentence-{num_sentence}_adultsVSados_smooth-{smoothing}.png')
