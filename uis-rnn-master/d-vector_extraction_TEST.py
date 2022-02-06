from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import pandas as pd
import os
import numpy as np
from tempfile import TemporaryFile

#Set directory path
os.chdir(Path("C:/Users/felix/OneDrive/Dokumente/Python Projects/audio_percentage"))

#transfer .csv-file into data frame
csv_folder = Path("C:/Users/felix/OneDrive/Dokumente/Python Projects/audio_percentage/audiosamples/output_documentation_TEST.csv")

df=pd.read_csv(csv_folder, sep=',',header=None)
#df_new= df.rename(columns={'0': 'audio_sample', '1': 'file_name','2': 'speaker_ID','3': 'file_path', '4': 'file_duration'  })

utterance_array1 = np.empty((0,256), dtype=float, order='C')
utterance_array2 = np.empty((0,1), dtype=float, order='C')

#create d-vector for each file (for each row in the csv-table)
def create_file_utterance(file_path):
    #give the file path to your audio file (must be "wav")

    # uses a VAD to trim out the silences in the audio file and also normalizes the decibel level of audio
    wav = preprocess_wav(file_path)
    encoder = VoiceEncoder("cpu") #encodes voice via cpu

    #embed_utterance takes in the processed wav file, segments it out into windows,
    # makes MFCCs of these segments and eventually creates d-vectors of these audio segments
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
    return cont_embeds

    #cont_embeds is a N by D matrix, where N is the number of segments created (=# of d-vectors)
    # and D is the dimension of each d-vector, which by default is 256

    #wav_splits is a list with the start and end time of each window for which a d-vector has been created.
    # -> number/sampling rate (=16000) = number in seconds (does not contain trimmed out silence)

#ITERATE DATA PATH FROM EACH SAMPLE
def iterate():
    global utterance_array1
    global utterance_array2
    for index, row in df.iterrows():
        file_path = Path(row[3]+ "/" + row[1]+".wav")
        utterance = create_file_utterance(file_path)

        #Creating table containing the utterance (d-vector), audioSuperCutNumber (consisting of n files) and speakerLabel
        #print(createTable(utterance, row[0], row[2]).shape)
        #utterance_array = np.vstack((utterance_array, createTable(utterance, row[0], row[2])))
        utterance_array1 = np.vstack((utterance_array1, utterance))

        sampleIndex = str(row[2])+"_"+str(row[0])
        r,c = utterance.shape


        utterance_array2 = np.vstack((utterance_array2, np.full((r, 1), sampleIndex)))



#merge table consisting of d-vectors & labels for all files together
def createTable(utterance, fileNumber, label):

    utt_rows, utt_cols = utterance.shape

    labelArray = np.full((utt_rows,1), label)
    fileNumberArray = np.full((utt_rows,1), fileNumber)

    print(fileNumber)
    print(label)
    print(utterance.shape)
    utterance = np.append(utterance, labelArray, axis=1)
    utterance = np.append(utterance, fileNumberArray, axis=1)

    return utterance

iterate()
print(utterance_array1.shape)
a,b = utterance_array2.shape

cluster_ID_array = utterance_array2.reshape((a,))
print(cluster_ID_array.shape)

# Convert features and corresponding classification labels into numpy arrays
outfile = TemporaryFile()
np.savez_compressed('./uis-rnn-master/d-vector_file_TEST', utterance_array_TEST=utterance_array1, cluster_ID_array_TEST=cluster_ID_array)

