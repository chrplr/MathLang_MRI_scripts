import os

import json
import pandas as pd
import numpy as np
import wave
import contextlib


###############################################################################
# Parameters
###############################################################################

# Directories
os.chdir('/neurospin/unicog/protocols/IRMf/MathLangage_Dehaene_Houenou_Moreno_2019/sentence_models/scripts')
datadir = "../../" + \
          "experiments_pc_stim/pc_stim_3T_2021-11-10/MathLangage"

onsetsdir = "../onsets"
data_dir = os.path.join(onsetsdir, "xpd_files")
stim_dir = os.path.join(onsetsdir, "theoretical_stims")


# Expyriment related
CLICK_DURATION = 100
TEXT_DURATION = 350
CUE_AUDIO_DURATION = 109   # get_audio_duration("bip_court.wav")
CUE_VIDEO_DURATION = 350
BLANK_DURATION = 1000

mri_key = "116"

key_trad = {"98": "r_click", "121": "l_click"}

cond_trad = {"control": "control_audio",
             "control0": "control_video",
             "colorlessg": "colorlessg_audio",
             "colorlessg0": "colorlessg_video",
             "general": "general_audio",
             "general0": "general_video",
             "tom": "tom_audio",
             "tom0": "tom_video",
             "context": "context_audio",
             "context0": "context_video",
             "arithprin": "arithprin_audio",
             "arithprin0": "arithprin_video",
             "arithfact": "arithfact_audio",
             "arithfact0": "arithfact_video",
             "geomfact": "geomfact_audio",
             "geomfact0": "geomfact_video",
             "bip": "cue_audio"}

# Exception subj : 102 and subj : 103 --> keypress did not work, must add two rows keypressed 98 and keypressed 121
dict_3 = {'102' : '200314', '103' : '120137'}

dict_name_stim = {'1' : '01', '2' : '02', '3': '03', '4' : '04', '5' : '05', '6' : '06', '7' : '07', '8' : '08', '9' : '09', '10' : '10', '11' : '11', '12' : '12', '13' : '13', '14' : '14', '15' : '15', '16' : '16', '50' : '15', '51' : '15', '52' : '15', '53' : '15', '54' : '15', '55' : '15', '56' : '15', '58' : '15', '59' : '15', '60' : '15', '99' : '01','101': '17', '102' : '18', '103' : '19', '104' : '20', '105' : '21' ,'106' : '22', '107' : '23', '108' : '24', '109' : '25', '110' : '26','111' : '27','112' : '28','113' : '29', '114' : '30', '115' : '31', '201' : '23', '202' : '24'}

# Subject related
description = dict()
with open(os.path.join(onsetsdir,"subjects_ados.json"), 'r') as f:
    description = json.load(f)
#has both adults and ados


###############################################################################
# Duration function
###############################################################################

# Get Raw Data
def get_raw_data(subj, bloc):

    xpd_name = description[f"sub-{subj:02}"]["list_xpd_file"][bloc - 1]
    csv_name = f"stim_subject{dict_name_stim[str(subj)]}_bloc_{bloc}a.csv"

    data_file = os.path.join(data_dir, xpd_name)
    stim_file = os.path.join(stim_dir, csv_name)

    data_df = pd.read_csv(data_file, header=0, comment="#")
    data_df = data_df.rename(columns={'time': 'onset'})

    stim_columns = ["onset_th", "stype_th", "cond_th", "pm_th", "id_th"]
    stim_df = pd.read_csv(stim_file, names=stim_columns)
    
    if data_df['pm'].isnull().sum() == len(data_df.index):  #ados_subjects audiovis file "" introduced 3 columns instead of 7
        data_df_split = pd.DataFrame(data_df['cond'].str.split(',').fillna('[ ]').tolist())
        data_df = pd.concat([data_df,data_df_split], axis = 1)
        data_df = data_df.drop(['cond','pm','stype','id','target_time'], axis = 1)
        data_df = data_df.rename(columns={'time': 'onset', 0 : 'cond', 1 : 'pm', 2 : 'stype', 3 : 'id', 4 :'target_time'})
        data_df['target_time'] = data_df['target_time'].astype(str)
        stim_df['onset_th'] = stim_df['onset_th'].astype(str)
    
    if not '98' in data_df["pm"].values:    #subject 102 and 103 miss keypressed 98 and 121
        print('Absence 98 et 121 dans le data_df')
        dict_3_trad=f"{subj:02d}"
        nip_participant = dict_3[dict_3_trad]
        dict_1 = {'subject_id' : nip_participant, 'onset' : 2000, 'cond' : 'keypressed', 'pm': '98', 'stype': None, 'id': None, 'target_time' : None}
        dict_2 = {'subject_id' : nip_participant, 'onset' : 2010, 'cond' : 'keypressed', 'pm': '121', 'stype': None, 'id': None, 'target_time' : None}
        data_df_add_row = pd.DataFrame([dict_1])
        data_df_add_row_bis = pd.DataFrame([dict_2])
        data_df = pd.concat([data_df,data_df_add_row], ignore_index = True)
        data_df = pd.concat([data_df,data_df_add_row_bis], ignore_index = True)
        
    data_df = data_df.rename(columns={'time':'onset'})

    return data_df, stim_df


# Readability correction
def merge_dfs(data_df, stim_df):

    data_df = data_df.set_index("target_time")
    df = stim_df.join(data_df, on="onset_th", how="left")

    # drop duplicate row for rsvp
    df = df[df['id'] != 'blank']
    
    # Sanity check
    same_size = (df.shape[0] == stim_df.shape[0])
    if not same_size:
        print("Error : wrong amount stim")
        print(f"df.shape = {df.shape[0]}, stim_df.shape = {stim_df.shape[0]}")

    # re-order columns for easier check
    columns_order = ['onset_th', 'onset',
                     'stype_th', 'stype',
                     'cond_th', 'cond',
                     'pm_th', 'pm',
                     'id_th', 'id']
    df = df[columns_order]


    # Differentiate blank & cues from video
    if len(df.loc[df['stype_th'] == "rsvp", 'stype'].unique()) == 1:
        df.loc[df['stype_th'] == "rsvp", 'stype'] = "video"
        df.loc[df['stype_th'] == "rsvp", 'stype_th'] = "video"
    else:
        print("Error : mutiple rsvp stype")

    # Sanity check
    same_stype = df["stype_th"].equals(df["stype"])
    if not same_stype:
        print("Error : stype differs")
        print(df.loc[df.stype != df.stype_th, [
              'stype', 'stype_th', 'cond', 'cond_th']])

    # change names for more coherence during anlysis
    df['cond'] = df['cond'].map(cond_trad)

    if len(df.loc[df['stype'] == "text", 'cond'].unique()) == 1:
        df.loc[df['stype'] == "text", 'cond'] = "blank"
        df.loc[df['stype'] == "text", 'cond_th'] = "blank"
    else:
        print("Error : mutiple text cond")

    if len(df.loc[df['stype'] == "redtext", 'cond'].unique()) == 1:
        df.loc[df['stype'] == "redtext", 'cond'] = "cue_video"
        df.loc[df['stype'] == "redtext", 'cond_th'] = "cue_video"
    else:
        print("Error : mutiple redtext cond")

    if len(df.loc[df['cond'] == "cue_audio", 'stype'].unique()) == 1:
        df.loc[df['cond'] == "cue_audio", 'stype'] = "bip"
        df.loc[df['cond'] == "cue_audio", 'stype_th'] = "bip"
    else:
        print("Error : mutiple cue_audio stype")

    return df


###############################################################################
# Duration function
###############################################################################

# Compute durations
def get_audio_duration(stim):

    fname = os.path.join(stim_dir, stim)
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = int(1000 * frames / float(rate))

    return duration


def get_video_duration(stim):

    wlist = stim.split()
    duration = TEXT_DURATION * len(wlist)

    return duration


def unitary_duration(stim, stype):

    if stype == "sound":
        #duration = pd.read_csv(os.path.join(stim_dir,'wavfiles_durations.csv'))
        duration = get_audio_duration(stim)
    elif stype == "video":
        #duration = pd.read_csv(os.path.join(stim_dir,'visualstim_durations.tsv'), sep='\t')
        duration = get_video_duration(stim)
    elif stype == "bip":
        duration = CUE_AUDIO_DURATION
    elif stype == "redtext":
        duration = CUE_VIDEO_DURATION
    elif stype == "text":
        duration = BLANK_DURATION
    else:
        print(f"ERROR : Unknown stype {stype}")
        return

    return duration


###############################################################################
# Prepare onset df
###############################################################################

def get_extended_onset_df(df):
    onset_df = df[["cond", "onset", "stype", "id_th"]]
    onset_df = onset_df.rename(columns={'id_th': 'stim'})

    onset_df['duration'] = onset_df.apply(
        lambda x: unitary_duration(x.stim, x.stype),
        axis=1)

    return onset_df


# Add click info
def add_click_info(onset_df, data_df):
    click_df = data_df[data_df['cond'] == "keypressed"]
    click_df = click_df[click_df['pm'] != mri_key]
    # print(click_df['pm'].unique())
    click_df = click_df[['onset', 'cond', 'pm']]
    click_df = click_df.rename(columns={'pm': 'stim', 'cond': 'stype'})
    click_df["cond"] = click_df["stim"].map(key_trad)
    click_df["stim"] = click_df["cond"]

    # Add click durations
    click_df['duration'] = CLICK_DURATION

    # Merge click info
    #onset_df = onset_df.append(click_df)
    onset_df = pd.concat([onset_df, click_df])
    onset_df = onset_df.sort_values('onset', ignore_index=True)

    return onset_df

def match_sentence(df):
    match_sentence_name = "matching_sentences_unique_number_and_number_cat_rev.csv"
    match_sentence_file = os.path.join(stim_dir, match_sentence_name)
        
    #matching sentence to unique number
    match_sentence = np.array(pd.read_csv(match_sentence_file))
    stim_array = np.array(df)
    for i in range(stim_array.shape[0]):
        for j in range(match_sentence.shape[0]):
           if stim_array[i,0] == match_sentence[j,4] or stim_array[i,0] == match_sentence[j,1]:
               stim_array[i,0] = int(match_sentence[j,0]) 
        if stim_array[i,0] == 'bip_court.wav':
            stim_array[i,0]='cue_audio'
        if stim_array[i,0] == '+':
            stim_array[i,0]='cue_video'
    df=pd.DataFrame(stim_array)
    df = df.rename(columns={0:"trial_type", 1:"onset", 2:"duration"})
    
    return df



# Main Function
def get_onset_df(subj, bloc):

    data_df, stim_df = get_raw_data(subj, bloc)
    df = merge_dfs(data_df, stim_df)
    onset_df = get_extended_onset_df(df)
    onset_df = add_click_info(onset_df, data_df)
    df = onset_df[["stim", "onset", "duration"]]
    df = match_sentence(df)
    df.onset = df.onset / 1000
    df.duration = df.duration / 1000

    return df


###############################################################################
# Test
###############################################################################

# subj = 2
# bloc = 1

# Elementary
# data_df, stim_df = get_raw_data(subj, bloc)
# df = merge_dfs(data_df, stim_df)
# onset_df = get_extended_onset_df(df)
# onset_df = add_click_info(onset_df, data_df)


# Main function
# onset_df = get_onset_df(subj, bloc)
# print(onset_df[onset_df.trial_type == "control_audio"])
