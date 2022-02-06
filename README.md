# Speaker_partition_module
Analysis of voices based on the Mel-frequency band.\
Goal: Identification of voices speaking (diarization) and calculation of speech partition (in %).

## Methodology:
- Collect voice data
- Sample audio data of x speakers that talk y times to represent a round of people talking
- Annotate samples with labels and merge audio file
- Create train & test split of samples
- Train unsupervised clustering module to detect number of people
- Train supervised RNN classifier to determine who is speaking at time x

## Preprocessing
- Convert files to .wav [convertFlac2Wav.py](https://github.com/felixbastian/Speaker_partition_module/blob/main/convertFlac2Wav.py)
- Collect data via LibriSpeech voices library (audiofiles) [audio_manipulation02.py](https://github.com/felixbastian/Speaker_partition_module/blob/main/audio_manipulation02.py)
- Extract x random speakers with y audio samples per speaker
Result: Generated audio samples of length 30-60 seconds

## Feature extraction:
- Create mel-frequency spectrum for each audio file [feature_extraction.py](https://github.com/felixbastian/Speaker_partition_module/blob/main/feature_extraction.py)
- Define overlapping feature window for training

## Training:
- Implementation of google-diarizer module
- Training accuracy is only at 40 %

## Further activity
- Create own unsupervised clustering module
- Try out different libraries
