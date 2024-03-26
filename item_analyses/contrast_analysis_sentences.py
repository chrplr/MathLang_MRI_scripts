"""
This scripts enables individual analysis of the MathLanguage experiment
using nilearn
You will obtain a per subject pdf containing sentences contrast plotted
on a glass brain

Written by Manon Pietrantoni with Christophe Pallier's advice and Bosco Taddei's template
"""
import os
os.chdir('/neurospin/unicog/protocols/IRMf/MathLangage_Dehaene_Houenou_Moreno_2019/sentence_models/scripts')
import sys
import numpy as np
import pandas as pd
import json
import pickle
import nibabel as nib
import time
import glob
from nilearn.image import load_img
from nilearn.masking import compute_epi_mask
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
#from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.datasets import fetch_localizer_contrasts

from create_onset_file_Manon import get_onset_df
from joblib import Parallel, delayed 
from cProfile import Profile
from pstats import SortKey, Stats

###############################################################################
# General parameters
###############################################################################
#SUBJECTS = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115] #sujets adolescents control
#SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,50,51,52,53,54,55,56,58,59,60] #sujets adultes + IBC controls
#SUBJECTS = [99]
#SUBJECTS = [201,202] # ados Asperger

SUBJECTS = [1,2]
#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
#SUBJECTS = [103]
start = time.time()

# Directories
SOURCEDIR = "../../Subjects_bids/fmriprep/"
SAVEDIR = "../results"

# Analysis variables
include_multi_run = False
use_deriv = False
try_load = True
contrasts_type = "sentence"
N_DRIFT = 3
hrf_model = "spm"
pdffile = "MathLang_singleruns"


if use_deriv:
    hrf_model = f"{hrf_model} + derivative"
    pdffile = f"{pdffile}_derivModel"

# Testing variables
is_test = False
# test_keys = ["control_audio", "mathVSlang"]
test_keys = []
if is_test:
    pdffile = "test"
    contrasts_type = "sentence"
    SAVEDIR = "Test"

    if len(test_keys) == 0:
        contrasts_type = "sentence"
    elif len(test_keys) < 3:
        try_load = False


# Experiment parameters
n_run = 5
n_cond = 6
n_acomp = 6


# Deduce Run and Design matrix Parameters
runs = range(1 - include_multi_run, n_run + 1)
mvmt_cfnds = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
pca_cfnds = [f"a_comp_cor_{i:02d}" for i in range(6)]
brain_cfnds = ["csf", "white_matter"]
confounds = mvmt_cfnds + pca_cfnds + brain_cfnds
n_cfnd = len(confounds)
mat_size = (1 + use_deriv) * n_cond + n_cfnd + N_DRIFT + 1


# Plotting parameters
Z_threshold = 3.1
vmax = 7
vmin = 0
plot_matrix = False
plot_map = False


###############################################################################
# Define contrasts
###############################################################################

# Basic

Id = np.eye(mat_size)
sentence_target = Id[0,:]
other_sentences = Id[1,:]

r_click = Id[2, :]
l_click = Id[3, :]
cue_audio = Id[4, :]
cue_video = Id[5, :]

contrasts_sentence={"sentence_target": sentence_target,
                    "other_sentences": other_sentences,
                    "r_click": r_click,
                    "l_click": l_click,
                    "cue_audio": cue_audio,
                    "cue_video": cue_video}


# All
contrasts_all = {}
contrasts_all.update(contrasts_sentence)


# Test
contrasts_test = {key: contrasts_all[key] for key in test_keys}

# Select contrast
dico_contrast = {"sentence": contrasts_sentence,}
contrasts = dico_contrast[contrasts_type]


###############################################################################
# Useful fuctions
###############################################################################

def get_col_order(use_deriv=use_deriv):

    conds = list(contrasts_sentence)
    drift = ["drift_" + str(i + 1) for i in range(N_DRIFT)] + ["constant"]

    if use_deriv:
        d_conds = [cond + "_derivative" for cond in conds]
        conds += d_conds

    order = conds + confounds + drift

    return order

def compute_design_matrix(subj, run, sentence):

    # Files path
    # onsetfile = os.path.join(SOURCEDIR,
    #                          "bids_dataset",
    #                          f"sub-{subj:02d}",
    #                          "func",
    #                          f"onsets_subject{subj:02d}_mathlang{run}.dat")
    file_id = f"sub-{subj:02d}_task-mathlang_run-{run}"
    cfnd_suffix = "desc-confounds_regressors"
    cfndfile = os.path.join(SOURCEDIR,
                            "derivatives",
                            "fmri_prep",
                            "fmriprep",
                            f"sub-{subj:02d}",
                            "func",
                            f"{file_id}_{cfnd_suffix}.tsv")
    TRfile = os.path.join(SOURCEDIR,
                          "bids_dataset",
                          f"sub-{subj:02d}",
                          "func",
                          f"sub-{subj:02d}_task-mathlang_run-{run:02}_bold.json")

    if not os.path.isfile(cfndfile):
        print(f"y Error : {cfndfile} Not Found")

    target = get_onset_df(subj, run).iloc[[sentence]]
    other = get_onset_df(subj, run).drop([sentence], axis=0)
    target.iloc[[0],0] = 'sentence_target' 

    for other_sentences in range(other.shape[0]):
        if other.iloc[other_sentences,0]!= 'cue_audio' and other.iloc[other_sentences,0]!= 'cue_video':
            if other.iloc[other_sentences,0]!= 'r_click' and other.iloc[other_sentences,0]!= 'l_click':
                other.iloc[[other_sentences],0] = 'other_sentences'

    onsets = pd.concat([target, other], axis = 0).sort_index()
    cfnds = pd.read_csv(cfndfile,
                        delimiter="\t",
                        engine='python')
    cfnds = cfnds.filter(confounds)

    with open(TRfile) as f:
        TR = json.load(f)["RepetitionTime"]
    timeframes = TR * np.arange(cfnds.shape[0])

    # Design Matrix
    design_matrix = make_first_level_design_matrix(timeframes,
                                                   onsets,
                                                   add_regs=cfnds,
                                                   drift_model='polynomial',
                                                   drift_order=N_DRIFT,
                                                   hrf_model=hrf_model)
    col_order = get_col_order()
    design_matrix = design_matrix.reindex(columns=col_order)

    return design_matrix


def get_design_matrix(subj, run, sentence, try_load=try_load):

    matrixDir = os.path.join(SAVEDIR,
                             "design_matrix",
                             hrf_model)
    matrixFile = os.path.join(matrixDir,
                              f"sub-{subj:02d}_sentence-{get_onset_df(subj, run).iloc[sentence,0]}_run-{run}.csv")
    matrixImg = os.path.join(matrixDir,
                             f"sub-{subj:02d}_sentence-{get_onset_df(subj, run).iloc[sentence,0]}_run-{run}.png")

    if try_load and os.path.isfile(matrixFile):
        design_matrix = pd.read_csv(matrixFile, index_col=0)
    else:
        print(f"        - Computing matrix sub{subj:02d} sentence_{get_onset_df(subj, run).iloc[sentence,0]}  run_{run} {hrf_model}")
        design_matrix = compute_design_matrix(subj, run, sentence)
        print("        - Done")

        design_matrix.to_csv(matrixFile)
        plot_design_matrix(design_matrix, output_file=matrixImg)

    return(design_matrix)

# Fonction récursive avec run = 0 calcul analyse multi-run
def get_Xy(subj, run, sentence, try_load=try_load):
    # if run == 0:
    #     X = [] #design Matrix
    #     y = [] #Signal fonction

    #     # Check if more runs than run == 0
    #     if len(runs) <= 1:
    #         print("recursive Error: runs empty")

    #     for run in runs[1:]:
    #         X_run, y_run = get_Xy(subj, run, sentence, try_load=try_load)
    #         X += X_run
    #         y += y_run

        # Files path
    file_id = f"sub-{subj:02d}_task-mathlang_run-{run}"
    func_suffix = "space-MNI152NLin2009cAsym_desc-preproc_bold"
    mask_suffix = "space-MNI152NLin2009cAsym_desc-brain_mask_new_affine"
    funcfile = os.path.join(SOURCEDIR,
                            "derivatives",
                            "fmri_prep",
                            "fmriprep",
                            f"sub-{subj:02d}",
                            "func",
                            f"{file_id}_{func_suffix}.nii.gz")
    maskfile = os.path.join(SOURCEDIR,
                            "derivatives",
                            "fmri_prep",
                            "fmriprep",
                            f"sub-{subj:02d}",
                            "func",
                            f"{file_id}_{mask_suffix}.nii.gz")

    if not os.path.isfile(funcfile):
        print(f"y Error : {funcfile} Not Found")

    load = include_multi_run or try_load
    design_matrix = get_design_matrix(subj, run, sentence, try_load=load)

    X = [design_matrix]
    y = [funcfile]

    return X, y, maskfile


def get_model(subj, run, sentence):
    
    num_sentence = get_onset_df(subj, run).iloc[sentence,0]
    modelfile = os.path.join(SAVEDIR,
                             "model",
                             hrf_model,
                             f"sub{subj:02d}_sentence_{num_sentence}_run-{run}.glm")

    if try_load and os.path.isfile(modelfile):
        with open(modelfile, 'rb') as pickle_file:
            fmri_glm = pickle.load(pickle_file)
    else:
        print(f"    - Computing model sub-{subj:02d} sentence_{get_onset_df(subj, run).iloc[sentence,0]} run-{run} {hrf_model}")

        design_matrices, run_imgs, mask_imgs = get_Xy(subj, run, sentence)
        print(mask_imgs)
        # fmri_glm = FirstLevelModel()
        fmri_glm = FirstLevelModel(mask_img=mask_imgs, smoothing_fwhm=3)
        fmri_glm = fmri_glm.fit(run_imgs, design_matrices=design_matrices)
        print("    - Done")
        print("time= ",(time.time() - start))

        with open(modelfile, 'wb') as pickle_file:
            pickle.dump(fmri_glm, pickle_file)

    return(fmri_glm)


# Tests get_model
# subj = SUBJECTS[0]
# run = runs[0]
# fmri_glm = get_model(subj, run)


def get_map(subj, run, con_key, con_val, sentence): 
    num_sentence = get_onset_df(subj, run).iloc[sentence,0]
    mapfile = os.path.join(SAVEDIR,
                           "stat_maps",
                           hrf_model,
                           f"sub{subj:02d}_sentence_{num_sentence}_run-{run}_{con_key}.nii")

    if try_load and os.path.isfile(mapfile):
        stat_map = load_img(mapfile)
    else:
        print(f"- Computing map sub-{subj:02d} sentence_{num_sentence} run-{run} {hrf_model} {con_key}")

        fmri_glm = get_model(subj, run, sentence)
        stat_map = fmri_glm.compute_contrast(con_val,
                                             stat_type='t',
                                             output_type="z_score")
        print("- Done")
        nib.save(stat_map, mapfile)

    return stat_map


###############################################################################
# Main code
###############################################################################

def create_pdf(SUBJECTS, runs, contrasts):

    for subj in SUBJECTS:

        print(f"sub{subj:02d}=====================")

        #pdf = PdfPages(f'{pdffile}_sub{subj:02d}.pdf')

        #subPage = plt.figure(figsize=[21, 29.7])
        #subPage.clf()
        #subPage.text(s=f"sub{subj:02d}",
        #             x=0.5,
        #             y=0.5,
        #             size=120,
        #             ha="center")
        #pdf.savefig()
        #plt.close()

#        for con_key, con_val in contrasts.items():

#            print(f"Starting contrast {con_key}")

            #fig, axs = plt.subplots(len(runs), figsize=[21, 29.7])
            #fig.suptitle(f"sub{subj:02d} \n contrast = {con_key} ",
            #             size=32,
            #             y=0.95)
        con_key = 'sentence_target'
        con_val = contrasts['sentence_target']
        for run in runs:
            print(f"run={run}")

            for sentence in range(get_onset_df(subj, run).shape[0]):
                print(f"sent={sentence}")
                if get_onset_df(subj, run).iloc[sentence,0]!= 'cue_audio' and get_onset_df(subj, run).iloc[sentence,0]!= 'cue_video' and get_onset_df(subj, run).iloc[sentence,0]!= 'l_click' and get_onset_df(subj, run).iloc[sentence,0]!= 'r_click':

                    stat_map = get_map(subj, run, con_key, con_val, sentence)

                        # Plot related
                        #gb_title = f"sub{subj:02d}_gb_{con_key}_r{run}"
                        #gb_display = plot_glass_brain(stat_map,
                        #                      threshold=Z_threshold,
                        #                      plot_abs=False,
                        #                      display_mode="lyrz",
                        #                      vmin=vmin,
                        #                      vmax=vmax,
                        #                      colorbar=True,
                        #                      cmap="seismic",
                        #                      figure=fig,
                        #                      axes=axs[run - 1],
                        #                      title=gb_title)

                # display = plot_stat_map(stat_map,
                #                         threshold=Z_threshold)
                #                         # output_file=mapfile)

            #pdf.savefig(fig)
            #plt.close()

        #d = pdf.infodict()
        #d['Title'] = 'MathLang Single Subject Analysis'
        #d['Author'] = 'Bosco TADDEI'
        #d['Subject'] = 'Study inter run reproductibly of MathLang paradigm'
        #d['Keywords'] = 'MathLanguage'

        #pdf.close()


###############################################################################
# Verifs
###############################################################################

#subj = int(sys.argv[1])

#create_pdf([subj], runs, contrasts)
create_pdf(SUBJECTS, runs, contrasts)




###############################################################################
# SECOND LEVEL ANALYSIS
###############################################################################
# print(f"total=====================")
 
# pdf = PdfPages(f'{pdffile}_total_ados_aout.pdf')
 
# subPage = plt.figure(figsize=[21, 29.7])
# subPage.clf()
# subPage.text(s="total",
#                  x=0.5,
#                  y=0.5,
#                  size=120,
#                  ha="center")
# pdf.savefig()
# plt.close()
# math_sentences = list(range(1,81))+list(range(241,281))
# language_sentences = list(range(121,161))+list(range(201,241))+list(range(281,321))
# control_sentences = list(range(81,121))+list(range(161,201))

# glob.glob = 

# conname = math_sentences
# name = glob.glob('sub14_sentence_84_run-*')


#     second_level_input = [os.path.join(SAVEDIR, "stat_maps/spm",f"sub{subj:02d}_sentence_{num_sentence}_run-{run}_sentence_target.nii") 
#                                                         for num_sentence in conname]
    
    
    
#     design_matrix = pd.DataFrame([1] * len(second_level_input), columns=['intercept'])
     
#     ############################################################################
#   # Model specification and fit.
#     second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
#     second_level_model = second_level_model.fit(second_level_input,design_matrix=design_matrix)
 
#   ##########################################################################
#   # To estimate the :term:`contrast` is very simple. We can just provide the column
#   # name of the design matrix.
#     z_map = second_level_model.compute_contrast(output_type='z_score')
#     alpha = 0.05
#     map, threshold =threshold_stats_img(stat_img=z_map, mask_img=None, alpha=0.05,   height_control='fdr', cluster_threshold=0, two_sided=True)
#     mapfile = os.path.join(SAVEDIR,"stat_maps",hrf_model, f"total_{conname}_sub-{subj:02d}.nii")
#     print("- Done")
#     nib.save(map, mapfile)
     
#      # display = plot_glass_brain(
#      #     z_map, threshold=threshold, colorbar=True, display_mode='lyrz', plot_abs=False,
#      #     title=f'{conname} (Z>{threshold} pfdr<{alpha})')
#      # plt.show()
 
#      print(f"Starting contrast {conname}")
#from joblib import Parallel, delayed      fig, axs = plt.subplots(1, figsize=[21, 29.7])
#      fig.suptitle(f"total \n contrast = {conname} ", size=32,y=0.95)
     
#      stat_map = z_map
 
#      # Plot related
#      gb_title = f"total_gb_{conname}"
#      gb_display = plot_glass_brain(stat_map,
#                                              threshold=Z_threshold,
#                                              plot_abs=False,
#                                              display_mode="lyrz",
#                                              vmin=vmin,
#                                              vmax=vmax,
#                                              colorbar=True,
#                                              cmap="seismic",
#                                              figure=fig,
#                                              title=gb_title)
#              # display = plot_stat_map(stat_map,
#              #                         threshold=Z_threshold)
#              #                         # output_file=mapfile)
     
#      pdf.savefig(fig)
#      plt.close()
 
# d = pdf.infodict()
# d['Title'] = 'MathLang Single Subject Analysis'
# d['Author'] = 'Bosco TADDEI & Manon PIETRANTONI'
# d['Subject'] = 'Study second level model Nilearn compared to spm of MathLang paradigm'
# d['Keywords'] = 'MathLanguage'
 
# pdf.close()                
