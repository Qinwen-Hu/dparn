# -*- coding: utf-8 -*-
"""
Created on Fri Apr 8 17:00:25 2022

@author: Zhongshu.Hou

You can use the code, if needed, to create training set 
where audios are splited into equal-length segments.
"""
import soundfile as sf
import librosa
from tqdm import tqdm
import numpy as np
import argparse

def main(args):
    raw_clean_dir = args.raw_clean_dir
    saved_clean_dir = args.saved_clean_dir
    raw_noise_dir = args.raw_noise_dir
    saved_noise_dir = args.saved_noise_dir
    len_s = args.segment_length
    
    #----------------Process clean dataset-----------------------
    clean_list = librosa.util.find_files(raw_clean_dir,ext='wav')
    buffer = np.array([], dtype='int16')
    idx = 0
    fileid = 0
    for dir in tqdm(clean_list):
        x, sr = sf.read(dir, dtype='int16')
        buffer = np.concatenate([buffer, x]) 
        idx += 1
        if idx == 100: # stack audios into one buffer and split
            split_list = np.array_split(buffer[:(len(buffer)-len(buffer)%(int(len_s*sr)))], (len(buffer)-len(buffer)%(int(len_s*sr)))//(int(len_s*sr)))
            buffer = buffer[(len(buffer)-len(buffer)%(int(len_s*sr))):]
            for clean_split in tqdm(split_list):
                sf.write(saved_clean_dir +  '/' + str(fileid) + '.wav', clean_split, sr)
                fileid += 1
            idx = 0
    
    #----------------Process noise dataset-----------------------
    noise_list = librosa.util.find_files(raw_noise_dir,ext='wav')
    buffer = np.array([], dtype='int16')
    idx = 0
    fileid = 0
    for dir in tqdm(noise_list):
        x, sr = sf.read(dir, dtype='int16')
        buffer = np.concatenate([buffer, x]) 
        idx += 1
        if idx == 100: # stack audios into one buffer and split
            split_list = np.array_split(buffer[:(len(buffer)-len(buffer)%(int(len_s*sr)))], (len(buffer)-len(buffer)%(int(len_s*sr)))//(int(len_s*sr)))
            buffer = buffer[(len(buffer)-len(buffer)%(int(len_s*sr))):]
            for clean_split in tqdm(split_list):
                sf.write(saved_noise_dir +  '/' + str(fileid) + '.wav', clean_split, sr)
                fileid += 1
            idx = 0
    
    print('There are {} splited clean audios created'.format(len(librosa.util.find_files(saved_clean_dir, ext='wav'))))
    print('There are {} splited noise audios created'.format(len(librosa.util.find_files(saved_noise_dir, ext='wav'))))
    

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_clean_dir", required=True, 
                        help='Path to the dir containing raw clean speech audio clips to be splited')
    parser.add_argument("--saved_clean_dir", required=True, 
                        help='Path to the dir saving splited clean speech audio clips')
    parser.add_argument("--raw_noise_dir", required=True, 
                        help='Path to the dir containing raw noise audio clips to be splited')
    parser.add_argument("--saved_noise_dir", required=True, 
                        help='Path to the dir saving splited noise audio clips')
    parser.add_argument("--segment_length", required=True, 
                        help='Splited audio length in seconds')
    
    args = parser.parse_args()
    main(args)
