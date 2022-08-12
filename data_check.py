import os
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
import torch

from utils import load_wav_to_torch
from mel_processing import spectrogram_torch

"""
VITS 数据集检测工具by GitHub@luoyily
https://github.com/luoyily/MoeTTS
使用方法：
0. 复制此文件到你的VITS项目下(跟train.py放在一个目录)
1. 在filelists填写你的数据集路径
2. 填写是否为单说话人数据, 是:True,否:False
3. 运行
"""
# Config：
filelists = './filelists/train_filelist.txt'
# is_relative = True
is_single_speaker = False

# CSV 检测
csv_flag = False
f = open(filelists, 'r', encoding='utf-8')
lines = list(f.readlines())
col_num_dic = {'True': 2, 'False': 3}
error_cols = [(n, len(l.split('|'))) for n, l in enumerate(lines)
              if len(l.split('|')) != col_num_dic[str(is_single_speaker)]]
if len(error_cols) == 0:
    csv_flag = True
    print('[1/3]CSV 检查通过')
else:
    for n, l in error_cols:
        print(f'错误行数：{n},此行含有{l}列')

# 文件路径检测
file_path_flag = False
voices = []
for line in lines:
    voice = line.split('|')[0]
    if not os.path.exists(voice):
        print(f'缺少文件：{voice}')
    else:
        voices.append(voice)
if len(voices) == len(lines):
    file_path_flag = True
    print('[2/3]文件路径 检查通过')
# 格式及采样率检测
audio_pass_flag = True
for voice in voices:
    try:
        audio, sampling_rate = load_wav_to_torch(voice)
        if sampling_rate != 22050:
            print(f'音频采样率错误：{voice}')
    except:
        print(f'其他错误，请检查格式是否为标准wav：{voice}')
        audio_pass_flag = False
        continue
    # 频谱预计算
    try:
        audio_norm = audio / 32768
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, 1024, 22050, 256, 1024, center=False).cpu()
        spec = torch.squeeze(spec, 0)
    except:
        print(f'频谱计算错误，请检查音频长度是否异常：{voice}')
        audio_pass_flag = False
if audio_pass_flag:
    print('[3/3]音频文件 检查通过')
if csv_flag and file_path_flag and audio_pass_flag:
    print('数据集完全检查通过！')
