from pydub import AudioSegment
from pathlib import Path
import librosa
import os
import random
import numpy as np

audio_mainFolder = Path("data/LibriSpeech/dev-clean")

#define the number of sound files to be created
iterations = 3

#output arrays
joint_sample_array = np.empty((0,5), dtype=str, order='C')
audio_list = np.array(os.listdir(audio_mainFolder))
sample_array = np.empty((0,4), dtype=str, order='C')

def manipulate_data(interations):
    for i in range(iterations):
        #CHOOSE RANDOM NUMBER OF SPEAKERS AND ACCESS RANDOM AUDIO FILE
        number_of_speakers = random.randint(2, 4)
        merged_audio_files = np.array([])

        choose_file(number_of_speakers, i)



def choose_file(number_of_speakers, i):
    for x in range(number_of_speakers):
        #choose random speaker and random path to audio folder
        speaker = random.choice(audio_list)
        speakers_folder = np.array(os.listdir(Path(str(audio_mainFolder)+"/" +speaker)))
        random_speaker_folder = random.choice(speakers_folder)

        #Create varaiables to access file path of speaker
        speaker_audio_filePath = Path(str(audio_mainFolder)+"/" + speaker +"/" + random_speaker_folder)
        speaker_audio_files = np.array(os.listdir(speaker_audio_filePath))

        audio_files_array = np.array([])

        #only choose the files with .flac ending and insert them in the array
        for f in os.listdir(speaker_audio_filePath):
            name, ext = os.path.splitext(f)
            if ext == '.flac':
                audio_files_array = np.append(audio_files_array, [name])

        #determine how many files per speaker
        number_of_audioFiles = random.randint(2, 4)

        generate_sample_array(speaker, number_of_audioFiles, audio_files_array, speaker_audio_filePath, i)

def generate_sample_array(speaker, number_of_audioFiles, audio_files_array, speaker_audio_filePath, i):
        random_audio_file = np.array([])
        for y in range(number_of_audioFiles):
            sample_array = np.append(sample_array,
                                     [[i,random.choice(audio_files_array), speaker, str(speaker_audio_filePath)]], axis=0)

        #shuffling the merged audio files twice for good randomness
        np.random.shuffle(sample_array)
        np.random.shuffle(sample_array)

        merge_clips(sample_array, i)

def merge_clips(sample_array, i):

    #MERGE AUDIOCLIPS FROM THE CHOSEN ONES (MIN X CLIPS; MAX Y) AND SAVE FIRST SAMPLE AS "A"
    #define an empty audiofile that can be appended (+=)
    audio = AudioSegment.empty()
    sample_length_array = np.empty((0,1), dtype=str, order='C')
    for x in range(len(sample_array)):

        #Define audio path and merge into one audio file
        audio_path = Path(sample_array[x,3] + "/" + sample_array[x,1]+ ".flac")
        audio += AudioSegment.from_file(audio_path, format="flac")

        #Add a column for sample length to the sample_array
        sample_length = librosa.get_duration(filename=audio_path)
        sample_length_array = np.append(sample_length_array, [sample_length])

    sample_length_array = sample_length_array.reshape((len(sample_array),1))
    sample_array = np.append(sample_array, sample_length_array, axis = 1)

    file_handle = audio.export(Path("audiosamples/output"+str(i)+".wav"), format="wav")
    joint_sample_array = np.concatenate((joint_sample_array, sample_array))


print(sample_array)
print(sample_array.shape)

print(joint_sample_array)
print(joint_sample_array.shape)
np.savetxt(Path("audiosamples/output_documentation.csv"), joint_sample_array, delimiter=",", fmt='%s')