import string
from pathlib import Path
from translate.translate_ds import TranslateUtil
from features import read_split_df
# Depends on nlp-util in PYTHONPATH

# Set working directory: cd /d D:\Dekel\Data\NLP\EntityMatching\ditto\preprocess

def read_split_df_first_letter_upper_case(path):
     df = read_split_df(path)
     df['tokens_str'] = df['tokens_str'].apply(lambda x: string.capwords(x))
     df['m_tokens_str'] = df['m_tokens_str'].apply(lambda x: string.capwords(x))
     return df
     
def translate_splits(folder_path, file_names):
     trutil = TranslateUtil(folder_path, file_names, tr_cols = ['tokens_str','m_tokens_str'], copy_cols = ['gt_label'], read_df_func = read_split_df_first_letter_upper_case)
     # 1) Save .xlsx files with eng, ready to upload
     trutil.save_eng_xls()
     # 2) Manually upload to Google Translate , the .xlsx files from eng_for_tr folder created in prev step --> save translations to ar
     # 3) create final files with column names in eng + copied columns from eng .xlsx
     trutil.process_tr_files(Path(folder_path) / 'ar')



if __name__=="__main__":
     DATASET_PATH = '../data/wiki/PER_12K_Aug_First_Last'
     translate_splits(folder_path = DATASET_PATH, file_names = ['train.txt','valid.txt','test.txt', 'per_mid_letter.txt', 'per_mid_name.txt'])
