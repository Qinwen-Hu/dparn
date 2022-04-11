# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:28:12 2022

@author: Zhongshu.Hou

Dataloeader 
"""

import soundfile as sf
import librosa
import torch
import numpy as np
from scipy import signal
import pandas as pd

'''
Input .csv file and path to rir audio clip
'''
TRAIN_CLEAN_CSV = '.csv'
TRAIN_NOISE_CSV = '.csv'
VALID_CLEAN_CSV = '.csv'
VALID_NOISE_CSV = '.csv'
RIR_DIR = ''

#-----------------set maximum T60 for rir-----------------------
T = int(500 * 48000 / 1000) #T60最大值500ms
t = np.arange(48000)
h = np.exp(-6 * np.log(10) * t / T)

#------simulated low-pass distortion to full-band signal--------
FIR_LOW = []
for cut_freq in range(16, 40):
    fir = signal.firwin(128, cut_freq /48.0)
    FIR_LOW.append(fir)

#----------------add simulated reverberation to signal----------
def add_pyreverb(clean_speech, rir):
    max_index = np.argmax(np.abs(rir))
    rir = rir[max_index:]
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[: clean_speech.shape[0]]

    return reverb_speech

#-----------------------mix speech and noise--------------------
def mk_mixture(s1,s2,snr,eps = 1e-8):    
    norm_sig1 = s1
    norm_sig2 = s2 * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(-snr/20)
    mix = norm_sig1 + alpha * norm_sig2
    M = max(np.max(abs(mix)),np.max(abs(norm_sig1)),np.max(abs(alpha*norm_sig2))) + eps
    if M > 1.0:    
        mix = mix / M
        norm_sig1 = norm_sig1 / M
        norm_sig2 = norm_sig2 / M

    return norm_sig1,mix

#-----------------------Dataset to load speech pair--------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, fs=48000, length_in_seconds=8, random_start_point=False, train=True):
        '''
        fs:                 sampling rate
        length_in_seconds:  audio length in seconds
        random_start_point: whether to read audio from the beginning
        train:              switch to train or validation mdoe
        '''
        self.train_clean_list = pd.read_csv(TRAIN_CLEAN_CSV)['file_dir'].to_list()
        self.train_noise_list = pd.read_csv(TRAIN_NOISE_CSV)['file_dir'].to_list()
        self.valid_clean_list = pd.read_csv(VALID_CLEAN_CSV)['file_dir'].to_list()[:5000]
        self.valid_noise_list = pd.read_csv(VALID_NOISE_CSV)['file_dir'].to_list()
        self.train_snr_list = pd.read_csv(TRAIN_CLEAN_CSV)['snr'].to_list()
        self.valid_snr_list = pd.read_csv(VALID_CLEAN_CSV)['snr'].to_list()
        self.L = length_in_seconds * fs
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.train = train
        self.rir_list = librosa.util.find_files(RIR_DIR,ext = 'wav')
        print('%s audios for training, %s for validation' %(len(self.train_clean_list), len(self.valid_clean_list)))

    def __getitem__(self, idx):
        #------------switch to train or validation mode-----------
        if self.train:
            clean_list = self.train_clean_list
            noise_list = self.train_noise_list
            snr_list = self.train_snr_list
        else:
            clean_list = self.valid_clean_list
            noise_list = self.valid_noise_list 
            snr_list = self.valid_snr_list   
            
        #------reverberattion, clipping and low-pass distortion rate-----------    
        reverb_rate = np.random.rand()
        clip_rate = np.random.rand()
        lowpass_rate = np.random.rand()

        #------read speech and noise-----------    
        if self.random_start_point:
            Begin_S = int(np.random.uniform(0,10 - self.length_in_seconds)) * self.fs
            Begin_N = int(np.random.uniform(0,10 - self.length_in_seconds)) * self.fs
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            noise, sr_n = sf.read(noise_list[idx % len(noise_list)], dtype='float32',start= Begin_N,stop = Begin_N + self.L)
        else:
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= 0,stop = self.L) 
            noise, sr_n = sf.read(noise_list[idx % len(noise_list)], dtype='float32',start= 0,stop = self.L)
            
        #-----------add simulated reverberation----------------------    
        if reverb_rate < 0.1: 
            rir_idx = np.random.randint(0,len(self.rir_list) - 1)
            rir_f = self.rir_list[rir_idx]
            rir_s = sf.read(rir_f,dtype = 'float32')[0]
            if len(rir_s.shape)>1:
                rir_s = rir_s[:,0]

            rir_s = rir_s[:min(len(h),len(rir_s))] * h[:min(len(h),len(rir_s))] 
            reverb = add_pyreverb(clean, rir_s)
        else:
            reverb = clean

        #--------------simulated low-pass distortion-----------------
        if lowpass_rate < 0.4: 
            id = np.random.randint(0,len(FIR_LOW))
            fir = FIR_LOW[id]
            reverb = np.convolve(reverb, fir)[127:127+len(reverb)]
            noise = np.convolve(noise, fir)[127:127+len(noise)]
    
        #--------------mixing-----------------    
        reverb_s,noisy_s = mk_mixture(reverb,noise,snr_list[idx],eps = 1e-8)

        #--------------simulated clipping distortion-----------------
        if clip_rate < 0.1: 
            noisy_s = noisy_s /np.max(np.abs(noisy_s) + 1e-12)
            noisy_s = noisy_s * np.random.uniform(1.2,3)
            noisy_s[noisy_s > 1.0] = 1.0
            noisy_s[noisy_s < -1.0] = -1.0
            reverb_s = reverb_s /np.max(np.abs(reverb_s) + 1e-12)
        
        return noisy_s.astype(np.float32), reverb_s.astype(np.float32)

    def __len__(self):
        if self.train:
            return len(self.train_clean_list)
        else:
            return len(self.valid_clean_list)

def collate_fn(batch):

    noisy, clean = zip(*batch)
    noisy = np.asarray(noisy)
    clean = np.asarray(clean)
    return noisy, clean 
