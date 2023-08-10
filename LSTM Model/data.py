import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from configuration import get_config
from utils import keyword_spot
import time

start = time.time()


config = get_config()   # get arguments from parser

# downloaded dataset path
#audio_path= r'VCTK-Corpus-0.92\wav48_silence_trimmed'                                          #utterance dataset
#clean_path = r'DS_10283_1942\clean_testset_wav\clean_testset_wav'  # clean dataset
#noisy_path = r'DS_10283_1942\noisy_testset_wav\noisy_testset_wav'  # noisy dataset

#D:\PROJECT\Speaker_Verification-master\Speaker_Verification-master\DS_10283_1942\Experiments

audio_path= r'VCTK-Corpus-0.92\wav48_silence_trimmed'                                          #utterance dataset
clean_path = r'DS_10283_1942\Experiments\clean_test'  # clean dataset
noisy_path = r'DS_10283_1942\Experiments\clean_noisy_test'  

import tensorflow as tf
import numpy as np
import librosa
import os

import tensorflow as tf

def extract_noise():
    # Define input layers for clean and noisy audio
    clean_input = tf.keras.layers.Input(shape=(None,))  # Shape: (batch_size, clean_audio_length)
    noisy_input = tf.keras.layers.Input(shape=(None,))  # Shape: (batch_size, noisy_audio_length)
    
    # Shared DNN layers to learn the mapping
    dnn_layers = tf.keras.Sequential([
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
    ])
    
    # Process clean and noisy audio through shared DNN layers
    clean_features = dnn_layers(clean_input)
    noisy_features = dnn_layers(noisy_input)
    
    # Compute the noise component by subtracting clean features from noisy features
    noise_component = tf.keras.layers.Subtract()([noisy_features, clean_features])
    
    # Define the model with clean and noisy audio as inputs and noise component as output
    model = tf.keras.Model(inputs=[clean_input, noisy_input], outputs=noise_component)
    
    return model


def save_spectrogram_tdsv():
    """ Select text specific utterance and perform STFT with the audio file.
        Audio spectrogram files are divided as train set and test set and saved as numpy file. 
        Need : utterance data set (VTCK)
    """
    print("start text dependent utterance selection")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    utterances_spec = []
    for folder in os.listdir(audio_path):
        spk_dir = os.path.join(audio_path, folder)
        spk_utts = os.listdir(spk_dir)
        spk_utts.sort()
        utter_path= os.path.join(spk_dir, spk_utts[0])
        if os.path.splitext(os.path.basename(utter_path))[0][-3:] != '001':  # if the text utterance doesn't exist pass
            print(os.path.basename(utter_path)[:4], "001 file doesn't exist")
            continue

        utter, sr = librosa.core.load(utter_path)
        sr=config.sr               # load the utterance audio
        utter_trim, index = librosa.effects.trim(utter, top_db=14)         # trim the beginning and end blank
        if utter_trim.shape[0]/sr <= config.hop*(config.tdsv_frame+2):     # if trimmed file is too short, then pass
            print(os.path.basename(utter_path), "voice trim fail")
            continue

        S = librosa.core.stft(y=utter_trim, n_fft=config.nfft,
                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))  # perform STFT
        S = keyword_spot(S)          # keyword spot (for now, just slice last 80 frames which contains "Call Stella")
        utterances_spec.append(S)    # make spectrograms list

    utterances_spec = np.array(utterances_spec)  # list to numpy array
    np.random.shuffle(utterances_spec)           # shuffle spectrogram (by person)
    total_num = utterances_spec.shape[0]
    train_num = (total_num//10)*9                # split total data 90% train and 10% test
    print("selection is end")
    print("total utterances number : %d"%total_num, ", shape : ", utterances_spec.shape)
    print("train : %d, test : %d"%(train_num, total_num- train_num))
    np.save(os.path.join(config.train_path, "train.npy"), utterances_spec[:train_num])  # save spectrogram as numpy file
    np.save(os.path.join(config.test_path, "test.npy"), utterances_spec[train_num:])


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth speaker processing..."%i)
        utterances_spec = []
        k=0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path) 
            sr=config.sr       # load utterance audio
            intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
            for interval in intervals:
                if (interval[1]-interval[0]) >= utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                    
                    if (interval[1]-interval[0]) > utter_min_len:
                      utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
                      utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance
                    else:
                      utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(config.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(config.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    extract_noise()
    if config.tdsv:
        save_spectrogram_tdsv()
    else:
        save_spectrogram_tisv()
    end=time.time()
    print("TOTAL TIME TAKEN TO RUN:",end-start)
