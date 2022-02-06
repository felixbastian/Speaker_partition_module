import os
from pydub import AudioSegment
from pathlib import Path

file_path = Path("C:/Users/felix/OneDrive/Dokumente/Python Projects/audio_percentage/data/LibriSpeech/dev-clean")

#Change working directory
os.chdir(file_path)

audio_files = os.listdir()

#iterate through folder structure
for a in os.listdir(file_path):
    first_sub_path = Path(str(file_path) + "/" + a)

    for f in os.listdir(first_sub_path):
        sub_path=Path(str(first_sub_path) + "/"+f)
        for s in os.listdir(sub_path):

            #extract the file extention for each file
            name, ext = os.path.splitext(s)

            #edit the file if .flac
            if ext == '.flac':
                sub_sub_path = Path(str(sub_path) + "/" + s)
                #create audio file (.wav)
                flac_tmp_audio_data = AudioSegment.from_file(sub_sub_path, sub_sub_path.suffix[1:])
                flac_tmp_audio_data.export(sub_sub_path.with_suffix(".wav"))

                #remove original .flac file
                os.remove(sub_sub_path)



