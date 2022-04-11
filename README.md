# MHA-DPCRN
We design a spectral compression mapping (SCM) for full-band speech enhancement, and propose a two-stage stream named MHA-DPCRN

# Rquirements
soundfile: 0.10.3  
librosa:   0.8.1  
torch:     3.7.10  
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
