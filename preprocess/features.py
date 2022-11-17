# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:53:32 2022

@author: family123
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd


def format_cand(cand_txt, add_chars = False):
    cand_flds = f"COL name VAL {cand_txt}"
    if add_chars:
        cand_flds += f" COL name_chars VAL {' '.join(list(cand_txt))}"
    return cand_flds

def write_split(folder_path, file_name, df, save_df = True, add_chars= False):    
    # save_ents_df(df, Path(folder_path) / (str(Path(file_name).stem) +  '.parquet'))        
    with open(Path(folder_path) / file_name, 'w', encoding = 'utf-8') as f:
        for rec_index, row in df.iterrows():
            f.write(f"{format_cand(row['tokens_str'], add_chars)}\t{format_cand(row['m_tokens_str'], add_chars)}\t{row['gt_label']}\n")


def read_split(txt_file_path):
    with open(txt_file_path, 'r', encoding="utf-8") as fl:
        rows = fl.readlines()
    return rows

def read_split_df(txt_file_path):
    rows = read_split(txt_file_path)
    tokens_str = []
    m_tokens_str = []
    gt_label = []
    
    for row in rows:
        flds = row.split('\t')
        cand1 = flds[0].split('COL name VAL ')[1]
        tokens_str.append(cand1)
        cand2 = flds[1].split('COL name VAL ')[1]
        m_tokens_str.append(cand2)
        gt_label.append(flds[2].strip())
    df = pd.DataFrame(data={'tokens_str': tokens_str, 'm_tokens_str': m_tokens_str, 'gt_label' :gt_label })    
    return df

def add_chars_split(folder_path, file_name):
    df = read_split_df(Path(folder_path) / file_name)
    write_split(Path(folder_path) / 'chars_aug', file_name, df, add_chars=True)
    
def add_chars_dataset(folder_path):    
    add_chars_split(folder_path, 'train.txt')
    add_chars_split(folder_path, 'valid.txt')
    add_chars_split(folder_path, 'test.txt')
    
if __name__=="__main__":
    add_chars_dataset( folder_path ='./data/per_bab' )    
