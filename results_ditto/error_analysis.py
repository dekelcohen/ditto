from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay

import sys
sys.path.insert(0,'../preprocess')
from features import read_split_df

ORG_75K_PATH = r'D:\NLP\Entity-Matching\hub\ditto\data\wiki\ORG_2_toks_75K'
ORG_PRED_FOLDER = r'test_predictions_train_4_to_1_test_7_to_1'

PER_12K_PATH = r'D:\NLP\Entity-Matching\hub\ditto\data\wiki\PER_12K'
PER_NEWS_PRED_FOLDER = r'predictions\train_wikidata_per_12k_test_news_f1_83'

GT_PATH = r'D:\NLP\Entity-Matching\hub\ditto\data\news\test\test_news_per.txt'
data_base_path = Path(PER_12K_PATH)

preds_json_path = data_base_path / PER_NEWS_PRED_FOLDER / 'output_small.jsonl'
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
    calc_metrics(df_pred) # PER_12K test_news_per.txt Precision 0.74, Recall 0.96, f1 0.837
    calc_metrics(df_test) # Classical: Precision 0.55, Recall 0.39, f1 0.45
    df_test[~df_test['match'] & (df_test.gt_label)][['tokens_str','m_tokens_str']].to_html('./temp/classical_fn.html')
    # Plot confusion matrix - many false-pos (almost no false-neg)
    ConfusionMatrixDisplay.from_predictions(df_pred.gt_label, df_pred['match'])
    # Examples of errors (FP): many unrelated pairs are predicted match=True    
    df_pred[df_pred['match'] & (~df_pred.gt_label)].to_html('./temp/ditto_fp_test_news_per.html')
    df_pred[df_pred['match'] & (df_pred.gt_label) & (df_pred.match_confidence < 0.85)].to_html('./temp/ditto_tp_low_confidence.html')
    
    
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