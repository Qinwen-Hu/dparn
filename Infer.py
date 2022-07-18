# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:05:17 2022

@author: Zhongshu.Hou

Modules
"""
import os
from Modules import DPModel
import torch
import soundfile as sf
import librosa 
from tqdm import tqdm
from signal_processing import iSTFT_module
from utils import is_pytorch_1_8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
WINDOW = torch.sqrt(torch.hann_window(1200,device=device) + 1e-8)
import argparse

#%%
def infer(args):
    '''model'''
    model = DPModel(model_type='DPARN', device=device) # 定义模型

    ''' load checkpoint'''
    checkpoint_DPARN = torch.load(args.check_dparn,map_location=device)

    '''load weights'''
    model_state_dict = checkpoint_DPARN['state_dict']
    if is_pytorch_1_8():
        # loading these weights crashes in PyTorch >=1.8
        model_state_dict.pop("process_model.intra_mha_list.0.MHA.out_proj.bias")
        model_state_dict.pop("process_model.intra_mha_list.1.MHA.out_proj.bias")
    model.load_state_dict(model_state_dict)

    model = model.to(device)
    model.eval()

    noisy_dir = args.noisy_dir
    noisy_list = librosa.util.find_files(noisy_dir, ext=['wav', 'flac', 'mp3'])
    os.makedirs(args.saved_enhanced_dir, exist_ok=True)

    with torch.no_grad():
        for noisy_f in tqdm(noisy_list):
            noisy_s = sf.read(noisy_f)[0].astype('float32')
            noisy_s = torch.from_numpy(noisy_s.reshape((1,len(noisy_s)))).to(device)
            noisy_stft = torch.stft(noisy_s,1200,600,win_length=1200,window=WINDOW,center=True)

            enh_stft = model(noisy_stft)
            enh_s = iSTFT_module(n_fft=1200, hop_length=600, win_length=1200,window=WINDOW,center = True)(enh_stft)
            enh_s = enh_s[0,:].cpu().detach().numpy()

            enh_s = librosa.resample(enh_s, 48000, 16000)

            sf.write(args.saved_enhanced_dir + '/' + noisy_f.split('/')[-1], enh_s, 16000)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_dparn", required=True, 
                        help='Path to DPARN checkpoints')
    parser.add_argument("--noisy_dir", required=True, 
                        help='Path to the dir containing noisy clips')
    parser.add_argument("--saved_enhanced_dir", required=True, 
                        help='Path to the dir saving enhanced clips')
    
    args = parser.parse_args()
    infer(args)

  