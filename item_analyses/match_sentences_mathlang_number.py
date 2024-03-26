##Match sentences mathlang with numbers##
import os
import numpy as np
import pandas as pd
import string

##Create matrix match unique number, sentence, category and category number
sourcedir=os.getcwd()
file= pd.read_csv(os.path.join(sourcedir,'stim_processing\mean_resp_rt_for_each_sentence_v02_sub-99_with_colors.csv'), sep=';', encoding='latin-1')
npfile=np.array(file)[:, [6,0,1]]
unique_number=np.array([range(1,321)])
npfile=np.concatenate((unique_number.T,npfile),axis=1)

##Add match audio file sentence
stim1a= pd.read_csv('stim_subject01_bloc_1a.csv', sep=',')
stim1b= pd.read_csv('stim_subject01_bloc_1b.csv', sep=',')
stim2a= pd.read_csv('stim_subject01_bloc_2a.csv', sep=',')
stim2b= pd.read_csv('stim_subject01_bloc_2b.csv', sep=',')
stim3a= pd.read_csv('stim_subject01_bloc_3a.csv', sep=',')
stim3b= pd.read_csv('stim_subject01_bloc_3b.csv', sep=',')
stim4a= pd.read_csv('stim_subject01_bloc_4a.csv', sep=',')
stim4b= pd.read_csv('stim_subject01_bloc_4b.csv', sep=',')
stim5a= pd.read_csv('stim_subject01_bloc_5a.csv', sep=',')
stim5b= pd.read_csv('stim_subject01_bloc_5b.csv', sep=',')
stim=np.vstack([stim1a, stim1b, stim2a, stim2b, stim3a, stim3b,stim4a, stim4b,stim5a, stim5b ])
npstim=np.array(stim)[:,[1,2,4]]

npstim_new=np.empty((0,1))
for i in range(npstim.shape[0]):
    if npstim[i,1]!='bip' and npstim[i,1]!='empty' and npstim[i,0]=='sound':
        npstim_new=np.vstack((npstim_new,npstim[i,2]))

npstim_match_audio=np.empty((0,4))
for i in range(npstim_new.shape[0]):
    npstim_match_audio=np.vstack((npstim_match_audio,npstim_new[i,0].split('_',3)))

###match à faire attention '2'=='02' donne False
npfile=np.hstack((npfile,np.zeros((320,1))))
for i in range(npfile.shape[0]):
    for j in range(npstim_match_audio.shape[0]):
        if '{:0>2}'.format(npfile[i,3])==npstim_match_audio[j,2] and npfile[i,2]==npstim_match_audio[j,1]:
            npfile[i,4]=npstim_new[j,0]

df = pd.DataFrame(data=npfile, columns=['unique_number','sentence','category','category_number','audio_file'])
df.to_excel('matching_sentences_unique_number_and_number_cat_verif.xlsx',index=False)
df.to_csv('matching_sentences_unique_number_and_number_cat_verif.csv',index=False)