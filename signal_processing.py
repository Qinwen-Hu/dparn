# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:05:31 2021

@author: xiaohuai.le
"""
import torch
from torch import nn
import numpy as np
import soundfile as sf
'''
STFT module for torch>=1.7
'''
class STFT_module(nn.Module):
     def __init__(self, n_fft, hop_length, win_length, center = True, normalized = False, window = torch.hann_window, mode = 'real_imag',device = 'cpu'):
         super(STFT_module, self).__init__()
         self.mode = mode
         self.n_fft = n_fft
         self.hop_length = hop_length
         self.win_length = win_length
         self.center = center
         self.normalized = normalized
         if not window:
             self.window = None
         else:
             #sinwin
             self.window = torch.sqrt( window(self.win_length)+1e-8).to(device)
         
     def forward(self, x):
         '''
         return: batchsize, 2, Time, Freq
         '''
         spec_complex = torch.stft(x, n_fft=self.n_fft, 
                                   hop_length=self.hop_length, 
                                   win_length=self.win_length,
                                   center=self.center,
                                   window=self.window,
                                   normalized=self.normalized,
                                   return_complex=False)
         if self.mode == 'real_imag':
             #return torch.permute(spec_complex,[0, 3, 2, 1])
             return spec_complex.permute([0, 3, 2, 1]).contiguous()
         elif self.mode == 'mag_pha':
             
             #spec_complex = torch.permute(spec_complex,[0, 3, 2, 1])
             spec_complex = spec_complex.permute([0, 3, 2, 1]).contiguous()
             mag = torch.sqrt(spec_complex[:, 0, :, :]**2 + spec_complex[:, 1, :, :]**2)
             angle = torch.atan2(spec_complex[:, 1, :, :],spec_complex[:, 0, :, :])
             return torch.stack([mag,angle],1)

'''
iSTFT module for torch >= 1.8
'''
class iSTFT_module_1_8(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, length, center = False, window = None, mode = 'real_imag',device = 'cpu'):
        super(iSTFT_module_1_8, self).__init__()
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
        self.center = center
        self.window = window
        if center:
            self.padding_num = int((self.win_length / 2 ) // (self.hop_length) * self.hop_length)
        else:
            self.padding_num = 0
            
        # if not window:
        #     self.window = torch.sqrt(torch.hann_window(self.win_length)+1e-8).to(device)
        
    def forward(self, x):
        '''
        return: batchsize, 2, Time, Freq
        '''
        length = self.win_length + self.hop_length * (x.shape[-2] - 1)
        spec_complex = x[:,0] + x[:,1] * 1j
        frame_chunks = torch.fft.irfft(spec_complex)
        frame_chunks = frame_chunks * self.window
        if self.center:
            s = torch.nn.Fold(output_size=[length,1], kernel_size=[self.win_length,1],stride=[self.hop_length,1])(frame_chunks.permute([0,2,1]))[:,0,self.padding_num:-self.padding_num,0] 
        else:
            s = torch.nn.Fold(output_size=[length,1], kernel_size=[self.win_length,1],stride=[self.hop_length,1])(frame_chunks.permute([0,2,1]))[:,0,:,0] 
        
        return s
    
'''
iSTFT module for torch <= 1.7
'''
class iSTFT_module_1_7(nn.Module):
    
    def __init__(self, n_fft, hop_length, win_length, length, center = False, window = None, mode = 'real_imag',device = 'cpu'):
        super(iSTFT_module_1_7, self).__init__()
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
        self.center = center
        if center:
            self.padding_num = int((self.win_length / 2 ) // (self.hop_length) * self.hop_length)
        else:
            self.padding_num = 0
            
        if not window:
            self.window = torch.sqrt(torch.hann_window(self.win_length)+1e-8).to(device)
        
    def forward(self, x):
        '''
        return: batchsize, 2, Time, Freq
        '''
        length = self.win_length + self.hop_length * (x.shape[-2] - 1)
        # bs,T,F,2 --> bs,T,F*2
        frame_chunks = torch.irfft(x.permute([0,2,3,1]),signal_ndim=1,signal_sizes=[self.win_length])
        frame_chunks = frame_chunks * self.window
        if self.center:
            s = torch.nn.Fold(output_size=[length,1], kernel_size=[self.win_length,1],stride=[self.hop_length,1])(frame_chunks.permute([0,2,1]))[:,0,self.padding_num:-self.padding_num,0] 
        else:
            s = torch.nn.Fold(output_size=[length,1], kernel_size=[self.win_length,1],stride=[self.hop_length,1])(frame_chunks.permute([0,2,1]))[:,0,:,0] 
        
        return s

def ola(inputs, win_size, win_shift):
    nframes = inputs.shape[-2]
    sig_len = (nframes - 1)* win_shift + win_size
    sig = np.zeros((sig_len,))
    ones = np.zeros((sig.shape))
    start = 0
    end = start + win_size
    for i in range(nframes):
        sig[start:end] += inputs[i, :]
        ones[start:end] += 1
        start = start + win_shift
        end= start + win_size
    return sig / ones

if __name__ == '__main__':
    '''
    if center, the STFT module will pad (win_length // 2 // hop_length) frames on both sides of the original sequence
    '''
    audio_test_dir = '/data/hdd0/zhongshu.hou/Torch_DPCRN/Valid_enh_FTCRN/noisy/nsy12.wav'
    au, _ = sf.read(audio_test_dir)
    # au = au[:-3]
    a = torch.from_numpy(au).reshape([1,len(au)])
    spec = STFT_module(n_fft=1200, hop_length=600, win_length=1200, center = True, normalized = False, window = torch.hann_window, )(a)
    s = iSTFT_module_1_8(n_fft=1200, hop_length=600, win_length=1200,center = True,length = a.shape[-1])(spec)

    print(torch.max(s - a))