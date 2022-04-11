# -*- coding: utf-8 -*-
"""
Created on Fri Apr 8 18:16:37 2022

@author: Zhongshu.Hou

Based on the clean and noise datasets, we creat.csv file to pair clean speech and noise,
also provided mixing SNR 
"""
import argparse
import librosa
from tqdm import tqdm
import numpy as np
import pandas as pd

def main(args):
    valid_rate = args.valid_rate
    SPEECH_DIR = args.clean_dataset_dir
    NOISE_DIR = args.noise_dataset_dir
    [snr_low, snr_high] = args.snr_range
    csv_dir = args.csv_saved_dir
    speech_list = []
    for dir in tqdm(SPEECH_DIR):
        lst = librosa.util.find_files(dir, ext='wav')
        print(len(lst))
        speech_list.extend(lst)
    
    noise_list = []
    for dir in tqdm(NOISE_DIR):
        lst = librosa.util.find_files(dir, ext='wav')
        print(len(lst))
        noise_list.extend(lst)
        
    np.random.shuffle(speech_list)
    np.random.shuffle(noise_list)
    print('There are {} clean speech clips andd {} noise clips'.format(len(speech_list),len(noise_list)))
     
    train_clean_list, valid_clean_list = speech_list[int(valid_rate * len(speech_list)):], speech_list[:int(valid_rate * len(speech_list))]
    train_noise_list, valid_noise_list = noise_list[int(valid_rate * len(noise_list)):], noise_list[:int(valid_rate * len(noise_list))]
    snr_train = np.random.randint(snr_low,snr_high,(len(train_clean_list))).tolist()  
    snr_valid = np.random.randint(snr_low,snr_high,(len(valid_clean_list))).tolist()

    train_clean_dict = {'file_dir':train_clean_list, 'snr':snr_train}
    valid_clean_dict = {'file_dir':valid_clean_list, 'snr':snr_valid}
    train_noise_dict = {'file_dir':train_noise_list}
    valid_noise_dict = {'file_dir':valid_noise_list}
    
    train_clean_data = pd.DataFrame(train_clean_dict);train_clean_data.to_csv(csv_dir + '/train_clean_data.csv')
    valid_clean_data = pd.DataFrame(valid_clean_dict);valid_clean_data.to_csv(csv_dir + '/valid_clean_data.csv')
    train_noise_data = pd.DataFrame(train_noise_dict);train_noise_data.to_csv(csv_dir + '/train_noise_dict.csv')
    valid_noise_data = pd.DataFrame(valid_noise_dict);valid_noise_data.to_csv(csv_dir + '/valid_noise_dict.csv')
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dataset_dir", required=True, 
                        help='Path to the dir of clean training dataset')
    parser.add_argument("--noise_dataset_dir", required=True, 
                        help='Path to the dir of clean training dataset')
    parser.add_argument("--valid_rate", required=True, default=0.08,
                        help='percentage of validation set')
    parser.add_argument("--snr_range", required=True, default=[-5,15],
                        help='[low limit, high limit]')
    parser.add_argument("--csv_saved_dir", required=True,
                        help='Path to save generated .csv file')
    
    args = parser.parse_args()
    main(args)