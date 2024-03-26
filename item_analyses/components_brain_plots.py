#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:44:18 2024

@author: mp271783
"""

import os.path as op
import matplotlib.pyplot as plt
from nilearn import surface
from nilearn import datasets
from nilearn import plotting

SAVEDIR = "../results/PCA_results" 
components = list(range(1,10))
#components = [1]
vmin = -3
vmax = 3
threshold = 0.8
#smoothing = 3.0

### Nilearn Slicing plots ###
for component in components:
    img = op.join(SAVEDIR,f"comp-{component}_PCA_in_29_participants.nii")
    plotting.plot_stat_map(img, display_mode='x',
                            cut_coords=range(-60, 70, 10), title=f'Slices_X_component_{component}', black_bg=True, threshold =threshold)
    plotting.plot_stat_map(img, display_mode='y',
                            cut_coords=range(-90, 80, 10), title=f'Slices_Y_component_{component}', black_bg=True, threshold =threshold)   
    plotting.plot_stat_map(img, display_mode='z',
                            cut_coords=range(-60, 80, 10), title=f'Slices_Z_component_{component}', black_bg=True, threshold =threshold)
    plotting.show()

### Nilearn Surface Plots ###
# cmap = 'cold_white_hot'
# for component in components:
#     print('computing component: ', component)
#     fsaverage = datasets.fetch_surf_fsaverage()
#     texture_right = surface.vol_to_surf(op.join(SAVEDIR,f"comp-{component}_PCA_in_29_participants.nii"), fsaverage.pial_right)
#     texture_left = surface.vol_to_surf(op.join(SAVEDIR,f"comp-{component}_PCA_in_29_participants.nii"), fsaverage.pial_left)
    
#     lateral_right=plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right',
#                                 title='Surface right hemisphere lateral view', colorbar=True, symmetric_cbar = "auto",
#                                 bg_map=fsaverage.sulc_right, cmap = cmap, threshold = threshold,vmin = vmin, vmax = vmax,
#                                 output_file = op.join(SAVEDIR,f"comp-{component}_PCA_normalized_in_29_participants_lateral_right_view.png"))
    
#     medial_right=plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right', view='medial',
#                                 title='Surface right hemisphere medial view', colorbar=True, symmetric_cbar = "auto",
#                                 bg_map=fsaverage.sulc_right, cmap = cmap,threshold = threshold,vmin = vmin, vmax = vmax,
#                                 output_file = op.join(SAVEDIR,f"comp-{component}_PCA_in_normalized_29_participants_medial_right_view.png"))
    
#     lateral_left=plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left',
#                                 title='Surface left hemisphere lateral view', colorbar=True, symmetric_cbar = 'auto',
#                                 bg_map=fsaverage.sulc_left, cmap = cmap,threshold = threshold,vmin = vmin, vmax = vmax,
#                                 output_file = op.join(SAVEDIR,f"comp-{component}_PCA_in_normalized_29_participants_lateral_left_view.png"))
    
#     medial_left=plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left', view='medial',
#                                 title='Surface left hemisphere medial view', symmetric_cbar = 'auto',
#                                 bg_map=fsaverage.sulc_left, cmap = cmap,threshold = threshold, vmin = vmin, vmax = vmax,
#                                 output_file = op.join(SAVEDIR,f"comp-{component}_PCA_in_normalized_29_participants_medial_left_view.png"))
      
#     plt.rcParams["font.size"] = 12
#     plt.rcParams["axes.titley"] = 0.9
#     plotting.show()                  
