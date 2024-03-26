#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:53:15 2024

@author: mp271783
"""
import numpy as np
import pandas as pd
import nibabel as nib
import os.path as op
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_img_comparison, plot_glass_brain
from nilearn import plotting

list_contrast_global = ['mathVSlang','colorlessgVScontrol','generalVScolorlessg','contextVSgeneral','tomVScontext','arithfactVSmath','arithprinVSmath','geomVSmath']
list_contrast_sentences = ['mathVSlang','meaninglessVSlist_word','generalVSmeaningless','contextualVSgeneral','socialVScontextual','calculusVSothermath','arithmeticVSothermath','geometryVSothermath']
mask = "mask_29_participants_intersection.nii"
maskfile = nib.load(mask)
masker = NiftiMasker(maskfile)

files_global = []
for contrast in list_contrast_global:
    print('computing contrast', contrast)
    global_model = op.join(f"../../iCortex_Bosco/oldstuff/Results_total/{contrast}_contrast_smooth8.0.nii")
    global_transformed = masker.fit_transform(global_model)
    files_global.append(global_transformed)
    
    
files_sentences = []
for contrast in list_contrast_sentences:
    print('computing contrast', contrast)
    sentence_model = op.join(f"../results/group_analysis_contrasts_sentence_model/{contrast}_contrast_smooth8.0.nii")
    sentence_transformed = masker.fit_transform(sentence_model)
    files_sentences.append(sentence_transformed)

for i in range(8):
    x = np.ravel(files_global[i])
    y = np.ravel(files_sentences[i])
    m, b = np.polyfit(x, y, 1)
    diff = y - m*x -b
    imdiff = masker.inverse_transform(diff)
    plot_glass_brain(imdiff, title = f'{list_contrast_sentences[i]}_contrast_global_modelVS_sentences_model',threshold=3.1121)
    plotting.show()
    plt.scatter(np.ravel(files_global[i]), np.ravel(files_sentences[i]), marker = '.',s=1)
    plt.title(f'{list_contrast_sentences[i]}_contrast_global_modelVS_sentences_model')
    plt.show()
