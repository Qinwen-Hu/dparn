# (Updating) DPARN
Official codes for the article:  *A light-weight full-band speech enhancement model*

DPARN is the acronym for dual-path attention -recurrent network.

We propose a spectral compression mapping (SCM) method to more effectively compress high-band spectral information, and utilize the multi-head attention mechanism to model the global spectral pattern.



# Experimental results

Results for ablation study on self-built dataset:

| Models    | PESQ     | STOI     | SI-SDR    |
| --------- | -------- | -------- | --------- |
| Noisy     | 1.45     | 0.90     | 5.00      |
| DPCRN     | 2.03     | 0.88     | 8.93      |
| DPCRN-SCM | 2.48     | 0.92     | 11.57     |
| DPARN-SCM | **2.65** | **0.93** | **12.56** |
| DPRAN-SCM | 2.31     | 0.92     | 10.68     |
| DPAAN-SCM | 2.10     | 0.91     | 10.29     |



Comparison with other full-band/super-wide-band speech enhancement model on VCTK-DEMAND dataset:

| Models                | Para. (M) | PESQ     | STOI     | SI-SDR    |
| --------------------- | --------- | -------- | -------- | --------- |
| Noisy                 | -         | 1.97     | 92.1     | 8.41      |
| RNNoise19 [2018]      | 0.06      | 2.34     | -        | -         |
| PerceptNet3 [2020]    | 8.0       | 2.73     | -        | -         |
| DeepFilterNet4 [2022] | 1.80      | 2.81     | -        | 16.63     |
| S-DCCRN5 [2022]       | 2.34      | 2.84     | 94.0     | -         |
| DPARN [2022]          | 0.89      | **2.92** | **94.2** | **18.28** |





# Requirements
soundfile: 0.10.3  
librosa:   0.8.1  
torch:     1.7.1  
numpy:     1.20.3  
scipy:     1.7.2  
pandas:    1.3.4  
tqdm:      4.62.3  



# Usage
1. Use Dataset_split.py to split audios to equal-length segments.  
2. Use Training_csv.py to generate .csv file to pair noise and clean speech  
3. Use Dataloader.py to create dataset iterater  
4. Set parameters in Modules.py  
5. Use Network_Training.py starting training and save checkpoints  
6. Use Infer.py to enhance noisy speech based on trained checkpoint.  
