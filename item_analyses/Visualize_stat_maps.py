import os
os.chdir('C:\\Users\\manon\\Documents\\école\\ENS\\Ecole\\2-Master1\\Stage\\Neurospin\\Script_Bosco\\iCortex_Bosco')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import nibabel as nib
from nilearn.plotting import plot_prob_atlas
from nilearn.decomposition import CanICA
from nilearn.image import load_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
##NilearnTEST##
from nilearn import plotting
img = "Results_total\\stat_maps\\spm\\total_test_aout_adosVSasp_tomVScontext.nii"
bg_img = "Results\\anat\\sub-51_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
plotting.plot_stat_map(img, display_mode='x',
                        cut_coords=range(-60, 70, 10), title='Slices_X_Nilearn', black_bg=True, threshold = 3.1, vmax = 10)
plotting.plot_stat_map(img, display_mode='y',
                        cut_coords=range(-90, 80, 10), title='Slices_Y_Nilearn', black_bg=True, threshold = 3.1, vmax = 10)      
plotting.plot_stat_map(img, display_mode='z',
                       cut_coords=range(-60, 80, 10), title='Slices_Z_Nilearn', black_bg=True, threshold = 3.1, vmax = 10)
                       

from nilearn import surface
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()
texture_right = surface.vol_to_surf("Results_total\\stat_maps\\spm\\total_test_aout_adosVSasp_tomVScontext.nii", fsaverage.pial_right)
texture_left = surface.vol_to_surf("Results_total\\stat_maps\\spm\\total_test_aout_adosVSasp_tomVScontext.nii", fsaverage.pial_left)
from nilearn import plotting

one=plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right',
                            title='Surface right hemisphere lateral view', colorbar=True,
                            threshold=3.1, bg_map=fsaverage.sulc_right, cmap = 'cold_white_hot', vmax = 10)

one_bis=plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right', view='medial',
                            title='Surface right hemisphere medial view', colorbar=True, symmetric_cbar = "auto",
                            threshold=3.1, bg_map=fsaverage.sulc_right, cmap = 'cold_white_hot', vmax = 10)

two=plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left',
                            title='Surface left hemisphere lateral view', colorbar=True, symmetric_cbar = 'auto',
                            threshold=3.1, bg_map=fsaverage.sulc_left, cmap = 'cold_white_hot',  vmax = 10)

two_bis=plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left', view='medial',
                            title='Surface left hemisphere medial view', symmetric_cbar = 'auto',
                            threshold=3.1, bg_map=fsaverage.sulc_left, cmap = 'cold_white_hot', vmax = 10)

plt.rcParams["font.family"] = "Times New Roman"   
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titley"] = 0.87
#plotting.show()                  

fsaverage = datasets.fetch_surf_fsaverage()
texture_right_math = surface.vol_to_surf("Results_total\\stat_maps\\spm\\total_adosVSadosAspergerarithprinVSmath_video.nii", fsaverage.pial_right)
texture_left_math = surface.vol_to_surf("Results_total\\stat_maps\\spm\\total_adosVSadosAspergerarithprinVSmath_video.nii", fsaverage.pial_left)

math_one=plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right_math, hemi='right',
                            title='Surface right hemisphere lateral view', colorbar=True,
                            threshold=3.1, bg_map=fsaverage.sulc_right, cmap = 'Blues', vmax = 10)

math_one_bis=plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right_math, hemi='right', view='medial',
                            title='Surface right hemisphere medial view', colorbar=True, symmetric_cbar = "auto",
                            threshold=3.1, bg_map=fsaverage.sulc_right, cmap = 'Blues', vmax = 10)

math_two=plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left_math, hemi='left',
                            title='Surface left hemisphere lateral view', colorbar=True, symmetric_cbar = 'auto',
                            threshold=3.1, bg_map=fsaverage.sulc_left, cmap = 'Blues',  vmax = 10)

math_two_bis=plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left_math, hemi='left', view='medial',
                            title='Surface left hemisphere medial view', symmetric_cbar = 'auto',
                            threshold=3.1, bg_map=fsaverage.sulc_left, cmap = 'Blues', vmax = 10)

plt.rcParams["font.family"] = "Times New Roman"   
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titley"] = 0.87
plotting.show() 
