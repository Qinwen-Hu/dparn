# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:32:17 2022

@author: Zhongshu.Hou & Qinwen.hu

Modules
"""
import torch
from torch import nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

'''
Import initialized SCM matrix
'''
Sc = np.load('./SpecCompress.npy').astype(np.float32)


class Encoder(nn.Module):
    
    def __init__(self, auto_encoder = True):
        super(Encoder, self).__init__()
        
        self.F = 601
        self.F_c = 256
        self.F_low = 125
        self.auto_encoder = auto_encoder

 
        self.flc_low = nn.Linear(self.F, self.F_low, bias=False)
        self.flc_high = nn.Linear(self.F, self.F_c - self.F_low, bias=False)

        self.conv_1 = nn.Conv2d(2,16,kernel_size=(2,5),stride=(1,2),padding=(1,1))
        self.bn_1 = nn.BatchNorm2d(16, eps=1e-8)
        self.act_1 = nn.PReLU(16)
        
        self.conv_2 = nn.Conv2d(16,32,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)
        
        self.conv_3 = nn.Conv2d(32,48,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_3 = nn.BatchNorm2d(48, eps=1e-8)
        self.act_3 = nn.PReLU(48)
        
        self.conv_4 = nn.Conv2d(48,64,kernel_size=(2,3),stride=(1,1),padding=(1,1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)
        
        self.conv_5 = nn.Conv2d(64,80,kernel_size=(1,2),stride=(1,1),padding=(0,1))
        self.bn_5 = nn.BatchNorm2d(80, eps=1e-8)
        self.act_5 = nn.PReLU(80)
    
    def init_load(self):
        self.flc_low.weight = nn.Parameter(torch.from_numpy(Sc[:self.F_low, :]), requires_grad=False)
        self.flc_high.weight = nn.Parameter(torch.from_numpy(Sc[self.F_low:, :]), requires_grad=True)
        
    def forward(self,x):
        #x.shape = (Bs, F, T, 2)
        x = x.permute(0,3,2,1) #(Bs, 2, T, F)
        x = x.to(torch.float32)
        x_low = self.flc_low(x)
        x_high = self.flc_high(x)
        x = torch.cat([x_low, x_high], -1)
        x_1 = self.act_1(self.bn_1(self.conv_1(x)[:,:,:-1,:]))
        x_2 = self.act_2(self.bn_2(self.conv_2(x_1)[:,:,:-1,:]))
        x_3 = self.act_3(self.bn_3(self.conv_3(x_2)[:,:,:-1,:]))
        x_4 = self.act_4(self.bn_4(self.conv_4(x_3)[:,:,:-1,:]))
        x_5 = self.act_5(self.bn_5(self.conv_5(x_4)[:,:,:,:-1]))    
        
        return [x_1,x_2,x_3,x_4,x_5]
    
    


class Real_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Real_Decoder, self).__init__()
        self.F = 601
        self.F_c = 256
        self.auto_encoder = auto_encoder

        self.real_dconv_1 = nn.ConvTranspose2d(160, 64, kernel_size=(1,2), stride=(1,1))
        self.real_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.real_act_1 = nn.PReLU(64)
        
        self.real_dconv_2 = nn.ConvTranspose2d(128, 48, kernel_size=(2,3), stride=(1,1))
        self.real_bn_2 = nn.BatchNorm2d(48, eps=1e-8)
        self.real_act_2 = nn.PReLU(48)
        
        self.real_dconv_3 = nn.ConvTranspose2d(96, 32, kernel_size=(2,3), stride=(1,1))
        self.real_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.real_act_3 = nn.PReLU(32)
        
        self.real_dconv_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2,3), stride=(1,1))
        self.real_bn_4 = nn.BatchNorm2d(16, eps=1e-8)
        self.real_act_4 = nn.PReLU(16)
        
        self.real_dconv_5 = nn.ConvTranspose2d(32, 1, kernel_size=(2,5), stride=(1,2))
        self.real_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.real_act_5 = nn.PReLU(1)

        self.inv_flc = nn.Linear(self.F_c, self.F, bias=False)
        
    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1 = self.real_act_1(self.real_bn_1(self.real_dconv_1(skipcon_1)[:,:,:,:-1]))
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2 = self.real_act_2(self.real_bn_2(self.real_dconv_2(skipcon_2)[:,:,:-1,:-2]))
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3 = self.real_act_3(self.real_bn_3(self.real_dconv_3(skipcon_3)[:,:,:-1,:-2]))
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4 = self.real_act_4(self.real_bn_4(self.real_dconv_4(skipcon_4)[:,:,:-1,:-2]))
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5 = self.real_act_5(self.real_bn_5(self.real_dconv_5(skipcon_5)[:,:,:-1,:-1]))              
        outp = self.inv_flc(x_5)
        return outp


class Imag_Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Imag_Decoder, self).__init__()

        self.F = 601
        self.F_c = 256
        self.auto_encoder = auto_encoder
        self.imag_dconv_1 = nn.ConvTranspose2d(160, 64, kernel_size=(1,2), stride=(1,1))
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)
        
        self.imag_dconv_2 = nn.ConvTranspose2d(128, 48, kernel_size=(2,3), stride=(1,1))
        self.imag_bn_2 = nn.BatchNorm2d(48, eps=1e-8)
        self.imag_act_2 = nn.PReLU(48)
        
        self.imag_dconv_3 = nn.ConvTranspose2d(96, 32, kernel_size=(2,3), stride=(1,1))
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)
        
        self.imag_dconv_4 = nn.ConvTranspose2d(64, 16, kernel_size=(2,3), stride=(1,1))
        self.imag_bn_4 = nn.BatchNorm2d(16, eps=1e-8)
        self.imag_act_4 = nn.PReLU(16)
        
        self.imag_dconv_5 = nn.ConvTranspose2d(32, 1, kernel_size=(2,5), stride=(1,2))
        self.imag_bn_5 = nn.BatchNorm2d(1, eps=1e-8)
        self.imag_act_5 = nn.PReLU(1)

        self.inv_flc = nn.Linear(self.F_c, self.F, bias=False)
        
    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4],dprnn_out],1)
        x_1 = self.imag_act_1(self.imag_bn_1(self.imag_dconv_1(skipcon_1)[:,:,:,:-1]))
        skipcon_2 = torch.cat([encoder_out[3],x_1],1)
        x_2 = self.imag_act_2(self.imag_bn_2(self.imag_dconv_2(skipcon_2)[:,:,:-1,:-2]))
        skipcon_3 = torch.cat([encoder_out[2],x_2],1)
        x_3 = self.imag_act_3(self.imag_bn_3(self.imag_dconv_3(skipcon_3)[:,:,:-1,:-2]))
        skipcon_4 = torch.cat([encoder_out[1],x_3],1)
        x_4 = self.imag_act_4(self.imag_bn_4(self.imag_dconv_4(skipcon_4)[:,:,:-1,:-2]))
        skipcon_5 = torch.cat([encoder_out[0],x_4],1)
        x_5 = self.imag_act_5(self.imag_bn_5(self.imag_dconv_5(skipcon_5)[:,:,:-1,:-1]))      
        outp = self.inv_flc(x_5)
        return outp

    
class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()
    
    

class DPARN(nn.Module):
    '''
    dual path, intra: MHAnet;  inter: RNN
    '''
    
    def __init__(self, numUnits, mha_blocks, n_heads, width, channel, device, **kwargs):
        super(DPARN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel       
        self.mha_blocks = mha_blocks
        self.d_model = numUnits
        self.d_ff = 4 * numUnits
        self.n_heads = n_heads
        self.device = device
        self.print = None
        
        self.pe = PositionalEncoding(input_size=self.d_model)
        
        self.intra_mha_list= nn.ModuleList([MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks)])
    
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8)

        self.inter_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)
        
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        
        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)


    
    def forward(self,x):
        # x.shape = (Bs, C, T, F)
        x = x.permute(0,2,3,1) #(Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()   
        ## Intra MHA    
        intra_MHA_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*T, F, C) 
        intra_MHA = intra_MHA_input + self.pe(intra_MHA_input)
        #attention block
        att_mask = AttentionMaskV2(causal=False)(intra_MHA).to(self.device)
        for att_block in self.intra_mha_list:
            intra_MHA = att_block(intra_MHA, att_mask) #(bs, L, d_model) = (Bs*T, F, C)
            
        intra_dense_out = self.intra_fc(intra_MHA) #(Bs*T, F, C)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel) #(Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = torch.add(x, intra_out)      
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0,2,1,3) #(Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1], inter_LSTM_input.shape[2], inter_LSTM_input.shape[3]) #(Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0,3,1,2)#(Bs, C, T, F)
        inter_out = inter_out.contiguous()
        
        return inter_out    

    
class DPRAN(nn.Module):
    '''
    dual path, intra: RNN;  inter: MHA
    '''
    
    def __init__(self, numUnits, mha_blocks, n_heads, width, channel, device, **kwargs):
        super(DPRAN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel       
        self.mha_blocks = mha_blocks
        self.d_model = numUnits
        self.d_ff = 4 * numUnits
        self.n_heads = n_heads
        self.device = device
        
        self.intra_rnn = nn.LSTM(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)
            
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8)
        
        self.pe = PositionalEncoding(self.d_model)

        self.inter_mha_list= nn.ModuleList([MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks)])
        
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        
        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)


    
    def forward(self,x):
        # x.shape = (Bs, C, T, F)
        x = x.permute(0,2,3,1) #(Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()  
        
        ## Intra RNN    
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0] #(Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel) #(Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = torch.add(x, intra_out)  
        
        
        ## Inter MHA
        inter_MHA_input = intra_out.permute(0,2,1,3).contiguous() #(Bs, F, T, C)
        inter_MHA_input = inter_MHA_input.view(inter_MHA_input.shape[0] * inter_MHA_input.shape[1], inter_MHA_input.shape[2], inter_MHA_input.shape[3]) #(Bs * F, T, C)
        inter_MHA = inter_MHA_input + self.pe(inter_MHA_input) #(Bs * F, T, C)        
        att_mask = AttentionMaskV2(causal=True)(inter_MHA).to(self.device) 
        for att_block in self.inter_mha_list:
            inter_MHA = att_block(inter_MHA, att_mask) #(bs, L, d_model) = (Bs*T, F, C)
        inter_dense_out = self.inter_fc(inter_MHA)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0,3,1,2)
        inter_out = inter_out.contiguous()
        
        return inter_out 
    
    
    
class DPAAN(nn.Module):
    '''
    dual path, intra: MHA;  inter: MHA
    '''
    
    def __init__(self, numUnits, mha_blocks, n_heads, width, channel, device, **kwargs):
        super(DPAAN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.width = width
        self.channel = channel       
        self.mha_blocks = mha_blocks
        self.d_model = numUnits
        self.d_ff = 4 * numUnits
        self.n_heads = n_heads
        self.device = device
        
        
        self.intra_mha_list = nn.ModuleList([MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks[0])])
    
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8)
        
        self.pe = PositionalEncoding(self.d_model)

        self.inter_mha_list= nn.ModuleList([MHAblockV2(self.d_model, self.d_ff, self.n_heads) for _ in range(self.mha_blocks[1])])
        
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        
        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)


    
    def forward(self,x):
        # x.shape = (Bs, C, T, F)
        x = x.permute(0,2,3,1) #(Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()  
        
        ## Intra MHA    
        intra_MHA_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) #(Bs*T, F, C) 
        intra_MHA = intra_MHA_input + self.pe(intra_MHA_input) #(Bs * F, T, C)  
        #attention block
        att_mask = AttentionMaskV2(causal=False)(intra_MHA).to(self.device)
        for att_block in self.intra_mha_list:
            intra_MHA = att_block(intra_MHA, att_mask) #(bs, L, d_model) = (Bs*T, F, C)
            
        intra_dense_out = self.intra_fc(intra_MHA) #(Bs*T, F, C)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel) #(Bs, T, F, C)
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = torch.add(x, intra_out) 
        
        ## Inter MHA
        inter_MHA_input = intra_out.permute(0,2,1,3).contiguous() #(Bs, F, T, C)
        inter_MHA_input = inter_MHA_input.view(inter_MHA_input.shape[0] * inter_MHA_input.shape[1], inter_MHA_input.shape[2], inter_MHA_input.shape[3]) #(Bs * F, T, C)
        inter_MHA = inter_MHA_input + self.pe(inter_MHA_input) #(Bs * F, T, C) 
        att_mask = AttentionMaskV2(causal=True)(inter_MHA).to(self.device)
        for att_block in self.inter_mha_list:
            inter_MHA = att_block(inter_MHA, att_mask) #(bs, L, d_model) = (Bs*T, F, C)
        inter_dense_out = self.inter_fc(inter_MHA)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0,3,1,2)
        inter_out = inter_out.contiguous()
        
        return inter_out    



class DPModel(nn.Module):
    '''
    Dual path model with encoder, decoder, processing block: mhanet or rnn
    '''
    #autoencoder = True
    def __init__(self, model_type, device):
        super(DPModel, self).__init__()
        self.device = device
        self.encoder = Encoder()
        self.model_type = model_type
        assert model_type in ['DPRAN', 'DPARN', 'DPAAN'], 'INVALIDE MODEL TYPE.'
        if self.model_type == 'DPARN':
            self.process_model = DPARN(numUnits=80, mha_blocks=2, n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPRAN':
            self.process_model = DPRAN(numUnits=80, mha_blocks=2, n_heads=8, width=127, channel=80, device=device)
        if self.model_type == 'DPAAN':
            self.process_model = DPAAN(numUnits=80, mha_blocks=[2, 2], n_heads=8, width=127, channel=80, device=device)            
        self.real_decoder = Real_Decoder()
        self.imag_decoder = Imag_Decoder()
        
    def init_load(self):
        self.encoder.init_load()
        
    def forward(self, x):
        # x --> audio batch
        # shape --> [Bs, sequence length]
        encoder_out = self.encoder(x) 
        dpath_out = self.process_model(encoder_out[4])
        enh_real = self.real_decoder(dpath_out, encoder_out)
        enh_imag = self.imag_decoder(dpath_out, encoder_out)
        enh_stft = torch.cat([enh_real, enh_imag], 1)#(Bs, 2, T, F)
        
        return enh_stft   