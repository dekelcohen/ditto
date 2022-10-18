from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay

data_base_path = Path(r'D:\NLP\Entity-Matching\hub\ditto\data\wiki\ORG_2_toks_75K')
preds_json_path = data_base_path / r'7_to_1_test_predictions\output_small.jsonl'
df_pred = pd.read_json(preds_json_path, lines=True)
df_pred['match'] = df_pred['match'].astype('bool')
df_pred['gt_label'] = df_test['gt_label'].to_numpy().astype('bool')

df_test = pd.read_parquet(data_base_path / r'dataframes\test.parquet')


def calc_metrics(df_pred):
    prec = precision_score(df_pred.gt_label, df_pred['match'])
    recall = recall_score(df_pred.gt_label, df_pred['match'])
    f1 = f1_score(df_pred.gt_label, df_pred['match'])
    print(f'Precision {prec}, Recall {recall}, f1 {f1}')    
    
    
    

if __name__=="__main__":
    df_pred['match'] = df_pred.match_confidence > 0.99 # Precision 0.878827790582812, Recall 0.9971978329908463, f1 0.9342784632887022
    calc_metrics(df_pred) # Precision 0.304, Recall 0.99, f1 0.466
    # Plot confusion matrix - many false-pos (almost no false-neg)
    ConfusionMatrixDisplay.from_predictions(df_pred.gt_label, df_pred['match'])
    # Examples of errors (FP): many unrelated pairs are predicted match=True    
    df_pred[df_pred['match'] & (~df_pred.gt_label)].to_html('./temp/ditto_fp_7_1_test.html')
    
    df_pred[df_pred['match'] & (~df_pred.gt_label)][['match_confidence']].describe(percentiles=np.arange(0, 1, 0.05))
    df_pred[df_pred['match'] & (df_pred.gt_label)][['match_confidence']].describe(percentiles=np.arange(0, 1, 0.05))




# df_pred = df_pred[df_pred.match_confidence > 0.99] # Precision 0.878827790582812, Recall 0.9971978329908463, f1 0.9342784632887022