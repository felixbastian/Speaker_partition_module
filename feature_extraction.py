import librosa
import os
import pandas as pd
import numpy as np
import math


from pathlib import Path  # dealing with paths

# defining the base folder & file (easier to deal with paths this way)
data_folder = Path("audiosamples")
audio_path = data_folder / "test.wav" #sample audio path


# number of files in target audio-file folder
# audio_list = os.listdir(data_folder)
# number_files = len(audio_list)
# print(number_files)

#transfer .csv-file into data frame

csv_folder = Path("audiosamples/output_documentation.csv")

df=pd.read_csv(csv_folder, sep=',',header=None)
#df_new= df.rename(columns={'0': 'audio_sample', '1': 'file_name','2': 'speaker_ID','3': 'file_path', '4': 'file_duration'  })

print(df)

#CREATE MFCC AND SPLIT IT INTO EQUALLY SIZED PIECE AND MERGE IT INTO A (40,) ARRAY

features = []

def extract_features(file_path,reference_row, class_label):
    audio, sample_rate = librosa.load(file_path)
    # Number of columns per second = (1 second samples ) * (sample rate = 22000) / (hop_length =500 by default) = 44 columns per second
    mfccs = librosa.feature.mfcc(y=audio, sr=22000, n_mfcc=40, hop_length = 500) #n_mfcc defines number of mfccs (dimensions)
    rows,columns = mfccs.shape
    number_of_mfcc_samples = math.ceil(columns/44)

    #split the array in seconds (44 samples per second) and calculate mean

    for x in range (number_of_mfcc_samples):

        if x*45 == columns | x*45 > columns:
            break
        a = x*44
        if(a+44 < columns):
            b = a+43
        else:
            b=columns

        mfccs_processed = np.mean(mfccs[0:40,a:b].T,axis = 0) #.T stands for


        features.append([mfccs_processed, reference_row, class_label])

        #debug data summary
        #print(str(mfccs_processed.shape) + str(x) + str(file_path) + "("+ str(a) +"---"+ str(b) + " of " + str(columns))


    #numberOfSeconds = (columns*500)/22000 #shows numbers of seconds of data file

    return mfccs_processed



#ITERATE DATA PATH FROM EACH SAMPLE
for index, row in df.iterrows():
    file_path = Path(row[3]+ "/" + row[1]+".wav")
    binary = row[5]
    extract_features(file_path, row, binary)

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature','reference_row', 'class_label'])
#pd.save('feature_extracted_df', featuresdf, delimiter=",")

featuresdf.to_csv(r'Testy.csv', index = False, sep=",")
#np.savetxt(Path("audiosamples/output_documentation.csv"), joint_sample_array, delimiter=",", fmt='%s')

print(featuresdf.shape)

#----------------------
#Everything IS working but the csv-file gets wrongly created!!
#----------------------

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
Y = np.array(featuresdf.class_label.tolist())
Z = np.array(featuresdf.reference_row.tolist())

np.save('test1', X)
np.save('test2', Y)
np.save('test3', Y)


#WRITE FUNCTION TO READ EACH MFCC PIECE AND MERGE IT INTO A (40,) ARRAY (LIKE THE FUNCTION FROM THE OTHER CASE)

#PUT ALL FILES INTO ONE ARRAY WITH THE RESPECTIVE 1/0 ELEMENT NEXT TO IT

#READY FOR TRAINING??


# #extract features
#
# #define function that takes file and returns the processed mfcc
# def extract_features(file_name):
#     audio, sample_rate = librosa.load(file_name)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) #n_mfcc defines number of mfccs (dimensions)
#     mfccs_processed = np.mean(mfccs.T, axis=0)#takes the mean for each dimension (normally it would be a (40,173) array for each mfcc
#
#     return mfccs_processed
#
# first_mfcc = extract_features(audio_path)
# print(first_mfcc)
# print(first_mfcc.shape)
#
# #defining librosa audio_path
# x, sr = librosa.load(audio_path)  # load. loads an audio file and decodes it into a 1-dim array
# print(type(x), type(sr))  # time series x and sampling rate sr (default = 22kHz)
#
# print(sr)
#
# # display waveform
# import matplotlib.pyplot as plt
# import librosa.display
#
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr) #librosa.display offers many options to display audio files
# #plt.show()
#
# #Display MFCC
# mfccs = librosa.feature.mfcc(x, sr=sr)
# #output length = (seconds = 9,81 for test file) * (sample rate = 22050 by default) / (hop_length =512 by default)
# # = 423 (always rounding up)
# print(mfccs)
# print(mfccs.shape)
# print(librosa.get_duration(filename=audio_path))
#
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# plt.show()


#features = []

# Iterate through each sound file and extract the features
# for index, row in df.iterrows():
#     file_name = str(data_folder) +"\\" + str(row['ID']) +'.wav'
#     #os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/', str(row["slice_file_name"])) #don't know what this is doing in source
#
#     data = extract_features(file_name)
#     class_label = row["Class"]
#
#     features.append([data, class_label])
#
# # Convert into a Panda dataframe
# featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
# pd.save('features_df', featuresdf)
#
# print(featuresdf)
#
# # Convert features and corresponding classification labels into numpy arrays
# X = np.array(featuresdf.feature.tolist())
# Y = np.array(featuresdf.class_label.tolist())
#
# np.save('feature_array', X)
# np.save('class_array', Y)


