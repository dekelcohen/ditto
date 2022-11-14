from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay

import sys
sys.path.insert(0,'../preprocess')
from features import read_split_df

DITTO_PATH = r'D:\Dekel\Data\NLP\EntityMatching\ditto' # r'D:\NLP\Entity-Matching\hub\ditto'
TEMP_PATH = Path(DITTO_PATH) / 'results_ditto/temp'
ORG_75K_PATH = Path(DITTO_PATH) / r'data\wiki\ORG_2_toks_75K'
ORG_PRED_FOLDER = r'test_predictions_train_4_to_1_test_7_to_1'

PER_12K_PATH = Path(DITTO_PATH) / r'data\wiki\PER_12K'
PER_NEWS_PRED_FOLDER = r'predictions\train_wikidata_per_12k_test_news_f1_83'

PER_12K_AUG_FIRST_LAST_PATH = Path(DITTO_PATH) / r'data\wiki\PER_12K_Aug_First_Last'
# PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_5000_CASED = r'predictions\train_wikidata_plus_5000_aug_first_last_neg_predict_on_news_per'
PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_500_UNCASED = r'predictions\train_wikidata_plus_500_aug_first_last_uncased_neg_predict_on_news_per'
PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_1000_UNCASED = r'predictions\train_wikidata_plus_1000_aug_first_last_uncased_neg_predict_on_news_per'
PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_2000_UNCASED = r'predictions\train_wikidata_plus_2000_aug_first_last_uncased_neg_predict_on_news_per'
PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_2000_UNCASED_FIX_LABELS = r'predictions\train_wikidata_plus_2000_aug_first_last_uncased_neg_predict_on_news_per_fix_labels'
NEWS_PER_ORG_PRED = r'predictions\train_10K_per_10K_org_predict_news_per_org'


PER_ORG_eq_amount_PATH = Path(DITTO_PATH) / r'data\wiki\PER_ORG_eq_amount'
NEWS_PER_ORG_PRED = r'predictions\train_10K_per_20K_org_predict_news_org'

GT_PATH = Path(DITTO_PATH) / r'data\news\test\test_news_org.txt'
data_base_path = Path(PER_ORG_eq_amount_PATH) # PER_12K_AUG_FIRST_LAST_PATH

preds_json_path = data_base_path / NEWS_PER_ORG_PRED / 'output_small.jsonl'
df_pred = pd.read_json(preds_json_path, lines=True)
df_pred['match'] = df_pred['match'].astype('bool')


df_test = read_split_df(GT_PATH)
# df_test = pd.read_parquet(data_base_path / r'dataframes\test.parquet')
df_pred['gt_label'] = df_test['gt_label'].astype('int32').to_numpy().astype('bool')




def calc_metrics(df_pred, gt_col = 'gt_label',pred_col = 'match'):
    prec = precision_score(df_pred[gt_col], df_pred[pred_col])
    recall = recall_score(df_pred[gt_col], df_pred[pred_col])
    f1 = f1_score(df_pred[gt_col], df_pred[pred_col])
    print(f'Precision {prec}, Recall {recall}, f1 {f1}')    
    
    
    

if __name__=="__main__":    
    calc_metrics(df_pred) 
    
    
    # PER_ORG_eq_amount 
      # train_10K_per_20K_org_predict_news_org Precision 0.75, Recall 0.96, f1 0.85
      # Conc
        # Very low prec on news ORGs, and the train includes 20K ORGS (worse with 57K ORGs) !
        # f1 0.915 on test_news_per_org.txt --> only because most are PER (90%)        
        # TODO: Threshold is too low (0.25 based on valid) - if th=0.71 -->
          # Precision 0.81, Recall 0.93, f1 0.87 (2 f1 points improvement)
        # ORG_2_toks_75K: better than news_orgs (results with 57K Orgs, not sure about 20K Orgs) test_f1 0.916, Precision 0.87, Recall 0.964
        # Adding all 57K ORGs in trainset doesn't help: test_f1 0.826, Precision 0.70 Recall 1.0
      
    # PER_12K (orig) test_news_per.txt - Precision 0.74, Recall 0.96, f1 0.837
    
    
    # PER_12K_Aug_First_Last PER+ORG: test_news_per_org.txt - Precision 0.86, Recall 0.97, f1 0.915 - low precision (but normal f1 score)
      # Conc: 
       # low precision 0.86 is unaccounted and can be improved 
       # FP: orgs adds to errors - since train doesn't include them - similar to new_per_only fp + some orgs (that were also fp before we deleted them from news_per ex: citibank group != softbank group - see fix_labels)
    
    # PER_12K_Aug_First_Last 2000 lower case char aug: test_f1 0.924, Precision 0.88, Recall 0.96 # small drop
    # PER_12K_Aug_First_Last 2000 lower case fix labels : test_f1 0.94, Precision 0.915, Recall 0.96
    # PER_12K_Aug_First_Last 2000 lower case (PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_2000_UNCASED): test_news_per.txt - Precision 0.89, Recall 0.91, f1 0.90
    # PER_12K_Aug_First_Last 1000 lower case (PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_1000_UNCASED): test_news_per.txt - Precision 0.84, Recall 0.92, f1 0.879
    # PER_12K_Aug_First_Last 500 lower case (PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_500_UNCASED): test_news_per.txt- Precision 0.79, Recall 0.92, f1 0.85
      # Conc: Adding 500 lower case negs first last (reoberta-base is cased) --> improves precision 0.74 --> 0.79, reduce recall 0.96 --> 0.92, improves f1 0.77 --> 0.85
    # Bug: Case: PER_12K_Aug_First_Last 5000 uppper case (PER_NEWS_PRED_FOLDER_AUG_FIRST_LAST_5000_CASED): test_news_per.txt- Precision 0.72, Recall 0.82, f1 0.77
      # Conc: Adding 5000 UPPER CASE negs first last  (reoberta-base is cased) --> reduce both precision and recall ! mainly recall
    # Classical on wikidata test: Precision 0.55, Recall 0.39, f1 0.45
    calc_metrics(df_test) 
    df_test[~df_test['match'] & (df_test.gt_label)][['tokens_str','m_tokens_str']].to_html(Path(TEMP_PATH) / 'classical_fn.html')
    # Plot confusion matrix - many false-pos (almost no false-neg)
    ConfusionMatrixDisplay.from_predictions(df_pred.gt_label, df_pred['match'])
    # Examples of errors (FP): many unrelated pairs are predicted match=True    
    # FP
    df_pred[df_pred['match'] & (~df_pred.gt_label)].to_html(Path(TEMP_PATH) / 'PER_ORG_eq_amount_20K_orgs_train_10K_PEr__f1_85_fp_test_news_org.html')
    # FN
    df_pred[~df_pred['match'] & (df_pred.gt_label)].to_html(Path(TEMP_PATH) / 'PER_ORG_eq_amount_20K_orgs_train_10K_PEr__f1_85_FN_test_news_org.html')
    # TP + TN
    df_pred[df_pred.match == df_pred.gt_label].to_html(Path(TEMP_PATH) / 'PER_ORG_eq_amount_20K_orgs_train_10K_PEr__f1_85_TP_TN_test_news__org.html')
    
    df_pred[df_pred['match'] & (df_pred.gt_label) & (df_pred.match_confidence < 0.85)].to_html('./temp/ditto_tp_low_confidence.html')
    
    # How many of match=True and gt_label=True are high confidence ?
    dt = df_pred[df_pred.match]
    dt[~dt.gt_label][['match_confidence']].describe(percentiles=np.arange(0, 1, 0.05))
    df_pc = df_pred.copy() 
    
    df_pc['match'] = df_pc.match & (df_pc.match_confidence > 0.71)
    calc_metrics(df_pc) 
    # Confidence distribtion for FP and TP    
    df_pred[df_pred['match'] & (~df_pred.gt_label)][['match_confidence']].describe(percentiles=np.arange(0, 1, 0.05))
    df_pred[df_pred['match'] & (df_pred.gt_label)][['match_confidence']].describe(percentiles=np.arange(0, 1, 0.05))
    
    # Hist: Ditto (train balanced, test 7:1 - already fixed to f1=0.93): Precision 0.304, Recall 0.99, f1 0.466

    ## EDA 
    # national in one and not in other - pos
    df_train[~df_train.tokens_str.str.contains(r'\bnational\b',regex=True) & df_train.gt_label & df_train.m_tokens_str.str.contains(r'\bnational\b',regex=True)][['tokens_str','m_tokens_str']]
    df_train = read_split_df(Path(PER_12K_PATH) / 'train.txt')
    df_train['gt_label'] = df_train['gt_label'].astype('int32').to_numpy().astype('bool')
    df_train[df_train.gt_label].to_html('./temp/train.html')
    
    # TODO: How many of train are eq ? Filter most of them
    


# df_pred = df_pred[df_pred.match_confidence > 0.99] # Precision 0.878827790582812, Recall 0.9971978329908463, f1 0.9342784632887022