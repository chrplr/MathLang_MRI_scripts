#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:11:39 2024

@author: mp271783
"""

import nibabel as nib
import os.path as op
from nilearn import surface
from nilearn import datasets
from nilearn import plotting
from nilearn.image import threshold_img
import matplotlib as plt

SAVEDIR = '../results'
math_sentences = list(range(1,81))+list(range(241,265))+list(range(266,281))
calculus_sentences = list(range(1,41))
arithmetic_sentences = list(range(41,81))
geometry_sentences = list(range(241,265))+list(range(266,281))

list_contrast = ['mathVSlang','colorlessgVScontrol','generalVScolorlessg','contextVSgeneral','tomVScontext','arithprinVSmath','arithfactVSmath','geomVSmath']

dict_contrast_math = {'calculusVSothermath':[calculus_sentences, arithmetic_sentences+geometry_sentences],
                 'arithmeticVSothermath':[arithmetic_sentences, calculus_sentences+geometry_sentences],
                 'geometryVSothermath':[geometry_sentences, calculus_sentences+arithmetic_sentences]}

threshold = 3.1285
for contrast in dict_contrast_math.keys():
    img = op.join(SAVEDIR,f"{contrast}_contrast_smooth-3.0.nii")
    threshold_value_img = threshold_img(img, threshold=threshold, copy=False)
    nib.save(threshold_value_img,f'../results/{contrast}_contrast_mask.nii')


'''plot the superposition in one image with Nilearn'''
mask_geometryVSothermath = nib.load('../results/geometryVSothermath_contrast_mask.nii')
mask_arithmeticVSothermath = nib.load('../results/arithmeticVSothermath_contrast_mask.nii')
mask_calculationVSothermath = nib.load('../results/calculusVSothermath_contrast_mask.nii')
math_masks = [mask_arithmeticVSothermath,mask_calculationVSothermath,mask_geometryVSothermath]

colors = ["blues","greens","reds"]
for image, strategy in zip(math_masks, ['calculusVSothermath', 'arithmeticVSothermath','geometryVSothermath']):
    title = (
        f"ROIs using {strategy} thresholding. "
        "Each ROI in same color is an extracted region"
    )
    plotting.plot_roi(image,title=title,)
plotting.show()

fsaverage = datasets.fetch_surf_fsaverage()
display_lateral_right=plotting.plot_surf_roi(fsaverage.infl_right, roi_map= mask_arithmeticVSothermath.get_fdata(),hemi='right',
                            title='Surface right hemisphere lateral view', colorbar=True,
                            threshold=threshold, bg_map=fsaverage.sulc_right,
                            output_file = f'../results/{contrast}_contrast_adultsVSados_lateral_right.png')

display_medial_right=plotting.plot_surf_roi(fsaverage.infl_right, roi_map= mask_arithmeticVSothermath.get_fdata(),hemi='right', view='medial',
                            title='Surface right hemisphere medial view', colorbar=True, symmetric_cbar = "auto",
                            threshold=threshold, bg_map=fsaverage.sulc_right, 
                            output_file = f'../results/{contrast}_contrast_adultsVSados_medial_right.png')

display_lateral_left=plotting.plot_surf_roi(fsaverage.infl_left,roi_map= mask_arithmeticVSothermath.get_fdata(), hemi='left',
                            title='Surface left hemisphere lateral view', colorbar=True, symmetric_cbar = 'auto',
                            threshold=threshold, bg_map=fsaverage.sulc_left, 
                            output_file = f'../results/{contrast}_contrast_adultsVSados_lateral_left.png')

display_medial_left=plotting.plot_surf_roi(fsaverage.infl_left, roi_map= mask_arithmeticVSothermath.get_fdata(),hemi='left', view='medial',
                            title='Surface left hemisphere medial view', symmetric_cbar = 'auto',
                            threshold=threshold, bg_map=fsaverage.sulc_left,
                            output_file = f'../results/{contrast}_contrast_adultsVSados_medial_left.png')
displays = [display_lateral_right,display_lateral_left, display_medial_left, display_medial_right]
for display in displays:
    for mask, color in zip(math_masks, colors):
        display.add_overlays(mask,threshold = threshold, colorbar = color, filled = True)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titley"] = 0.9
    plotting.show()      