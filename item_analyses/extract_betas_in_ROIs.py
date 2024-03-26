#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:08:25 2024

@author: mp271783
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os.path as op
import glob
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker
from nilearn.maskers import NiftiLabelsMasker
from nilearn.masking import apply_mask
from nilearn.plotting import plot_epi, plot_roi, show, plot_img, plot_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
from nilearn.connectome import ConnectivityMeasure

SENTENCE_MAPS = '../results/Group_analysis_individual_sentence'
num_all_sentences = list(range(1,265))+list(range(266,321))

#atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas = datasets.fetch_atlas_juelich("maxprob-thr0-1mm")
labels = atlas.labels

# Instantiate the masker with label image and label values
masker = NiftiLabelsMasker(
    atlas.maps,
    labels=atlas.labels,
    standardize="zscore_sample",
)

matrix_ROIs = np.empty((0,len(labels)-1))
for num_sentence in num_all_sentences:
    sentence = op.join(SENTENCE_MAPS, f"sentence_{num_sentence}_smooth8.0.nii")
    masker.fit(sentence)
    signals = masker.transform(sentence)
    matrix_ROIs=np.vstack((matrix_ROIs, signals))
print(matrix_ROIs.shape, type(matrix_ROIs))
correlation_measure = ConnectivityMeasure(kind="correlation")
correlation_matrix = correlation_measure.fit_transform([matrix_ROIs])[0]
np.fill_diagonal(correlation_matrix, 0)
plotting.plot_matrix(correlation_matrix,figure=(10, 8),labels=labels[1:],vmax=0.8,vmin=-0.8,title="Correlation in ROIs 319 sentences",reorder=True,)
plotting.show()

#df = pd.DataFrame(data=matrix_ROIs, columns = labels[1:])
#df.to_csv('matrix_sentencesXROIs.csv', index=False)