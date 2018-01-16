import re
import os
from glob import glob
import librosa
import numpy as np
import cv2
from config import LABELS



def list_wavs_fname(dirpath, ext='wav'):
    '''
    데이터 파일 읽어옴.
    '''
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def chop_audio(wav, L=16000, num=100, speed_tuning=True, tuning_proba=0.5):
    '''
    chop audios that are larger than 16000(eg. wav files in background noises folder) to 16000 in length.
    create several chunks out of one large wav files given the parameter 'num'.
    '''
    for i in range(num):
        if speed_tuning and np.random.random() > tuning_proba:
            speed_rate = np.random.uniform(0.7,1.3)
            _wav = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
        else:
            _wav = wav
        beg = np.random.randint(0, len(_wav) - L)
        yield _wav[beg: beg + L]

def pad_audio(wav, L=16000):
    '''
    pad audios that are less than 16000(1 second) with 0s to make them all have the same length.
    '''
    if len(wav) >= L: 
        return wav
    else: 
        return np.pad(wav, pad_width=(L - len(wav), 0), mode='constant', constant_values=(0, 0)) 
        # sample 앞뒤로 constant_values[0]과 constant_values[1]을 각각 pad_width 갯수 만큼 패딩
        # 총길이는 len(samples) + 2*pad_width
        

def log_specgram(wav, sr, eps=1e-8):
    '''
    로그 스펙트로그램 변환
    '''
    D = librosa.stft(wav, n_fft=240, hop_length=60, win_length=240, window='hamming')
    spect, phase = librosa.magphase(D)
    return np.log(spect + eps)

def mel_specgram(wav, sr, eps=1e-8):
    '''
    멜 스펙트로그램 변환
    '''
    M = librosa.feature.melspectrogram(wav, sr=sr, n_mels=200, hop_length=60, fmin=20, fmax=4000)
    melgram = librosa.logamplitude(M, ref_power=1.0)
    return melgram



def label_transform(labels):
    '''
    레이블 정규화 및 one-hot벡터화 (더미화)
    '''
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in LABELS:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    
    nlabels = [LABELS.index(l) for l in nlabels]
    return nlabels
