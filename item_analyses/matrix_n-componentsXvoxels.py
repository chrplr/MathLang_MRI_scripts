#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:58:44 2024

@author: mp271783
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain

data_components = np.array(pd.read_csv('../results/matrix_sentencesXpca_components.csv'))
data_voxels = np.array(pd.read_csv('sub-total_matrix_sentencesXvoxels.csv'))

print(data_components.T.shape, data_voxels.shape)

matrix_compXvoxels = np.dot(data_components.T,data_voxels)
print(matrix_compXvoxels.shape)
df = pd.DataFrame(data=matrix_compXvoxels)
df.to_csv('matrix_pca_componentsXvoxels.csv', index=False) 