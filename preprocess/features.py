# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:53:32 2022

@author: family123
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

folder_path = './temp'

def format_cand(cand_txt, add_chars = False):
    cand_flds = f"COL name VAL {cand_txt}"
    if add_chars_aug:
        cand_flds += f"\tCOL name_chars VAL {' '.join(list(cand_txt))}"
    return cand_flds

def write_split(file_name, df, add_chars= False):    
    save_ents_df(df, Path(folder_path) / (str(Path(file_name).stem) +  '.parquet'))        
    with open(Path(folder_path) / file_name, 'w', encoding = 'utf-8') as f:
        for rec_index, row in df.iterrows():
            f.write(f"{format_cand(row['tokens_str'], add_chars)}\t{format_cand(row['m_tokens_str'], add_chars)}\t{row['gt_label']}\n")

write_split('train.txt', df_train)
write_split('valid.txt', df_val)
write_split('test.txt', df_test)


def read_split(txt_file_path):
    with open(txt_file_path, 'r') as fl:
        rows = fl.readlines()
    return rows
    
def add_chars_split(txt_file_path, file_name):
    rows = read_split(Path(folder_path) / file_name)
    for row in rows:
        flds = rows[0].split('\t')
        cand1 = flds[0].split('COL name VAL ')[1]
        cand2 = flds[1].split('COL name VAL ')[1]
        gt_label = flds[2].strip()
        
    write_split(Path(folder_path) / 'chars_aug' / file_name)
    
def add_chars_dataset(folder_path):    
    add_chars_split(folder_path, 'test.txt')
    

add_chars_dataset( folder_path ='./data/per_bab' )    