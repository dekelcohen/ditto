from translate.translate_ds import TranslateUtil
from features import read_split_df

def translate_splits(folder_path, file_names = ['train.txt','valid.txt','test.txt']):
     trutil = TranslateUtil(folder_path, file_names,tr_cols = ['tokens_str','m_tokens_str'],copy_cols = ['gt_label'],read_df_func = read_split_df)
     trutil.save_eng_xls()  



if __name__=="__main__":
     DATASET_PATH = '../data/wiki/PER_12K_Aug_First_Last'
     translate_splits(folder_path = DATASET_PATH, file_names = ['train.txt','valid.txt','test.txt', 'per_mid_letter.txt', 'per_mid_name.txt'])
