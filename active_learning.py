from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data
from ditto_light.ditto import train, evaluate
from ditto_light.dataset import DittoDataset

from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner

# Pre-set our batch sampling to retrieve 3 samples at a time.
N_INITIAL_TRAIN_SAMPLES = 2000
N_AL_ADDITIONAL_SAMPLES = 10
N_AL_LABELING_QUERIES = 1 # >=1 : 0 causes div by zero 
QRY_BATCH_SIZE = N_AL_ADDITIONAL_SAMPLES // N_AL_LABELING_QUERIES
RS = 42

preset_uncertainty_sampling = partial(uncertainty_sampling, n_instances=QRY_BATCH_SIZE)

# sklearn style wrapper (fit, predict_proba, score) for Ditto train and predict code 
class DittoFitPredictWrapper:
    def __init__(self):
        pass
    
    def fit(self, X, y, train_dataset, valid_dataset, test_dataset, run_tag, hp):
        print(f'Enter fit type(X)={type(X)}, X.shape[0]={X.shape[0]}')
        al_trainset = DittoDataset(path=None, pairs=list(X), labels=list(y), lm=hp.lm, max_len=hp.max_len, size=hp.size, da=hp.da)
        
        self.model = train(al_trainset,
                valid_dataset,
                test_dataset,
                run_tag, hp)
        
    def predict_proba(self, X, **fit_kwargs):
        print(f'Enter predict_proba type(X)={type(X)}, X.shape[0]={X.shape[0]}')
        hp = fit_kwargs['hp']
        # X --> DittoDataset --> iterator (DataLoader) --> predict probas 
        dummy_labels = [0] * X.shape[0] # Predict doesn't need labels (y) - only X, but DittoDataset requires them
        unlabeled_dataset = DittoDataset(path=None, pairs=list(X), labels=dummy_labels, lm=hp.lm, max_len=hp.max_len, size=hp.size, da=hp.da)
        
        iterator = data.DataLoader(dataset=unlabeled_dataset,
                             batch_size=hp.batch_size*16,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=unlabeled_dataset.pad)        
        # TODO:Debug:Dummy code - np arr shapes 
        # all_probs = np.random.random((QRY_BATCH_SIZE * 3, 2))
        # all_probs[:,1] = 1 - all_probs[:,0]
        
        all_probs = np.empty((0,2))
        with torch.no_grad():
            for batch in iterator:
                x, y = batch
                logits = self.model(x)
                probs = logits.softmax(dim=1)
                all_probs = np.concatenate([all_probs,probs.cpu().numpy()]) 
        
        
        print(f'all_probs= {all_probs[:10]}')        
        return all_probs
    
    def score(self,dummy_X, dummy_y, **fit_kwargs):
        print(f'Enter score - does not use X and y but rather valid_dataset and test_dataset')
        test_dataset = fit_kwargs['test_dataset']
        valid_dataset = fit_kwargs['valid_dataset']
        
        hp = fit_kwargs['hp']
        
        padder = DittoDataset.pad
        
        test_iter = data.DataLoader(dataset=test_dataset,
                                     batch_size=hp.batch_size*16,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=padder)
        valid_iter = data.DataLoader(dataset=valid_dataset,
                                     batch_size=hp.batch_size*16,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=padder)
        
        
        self.model.eval()
        dev_f1, th = evaluate(self.model, valid_iter)
        test_f1 = evaluate(self.model, test_iter, threshold=th)
        return test_f1




# train_dataset, valid_dataset, test_dataset, run_tag, hp    
def al_train(**fit_kwargs ):
    sk_ditto = DittoFitPredictWrapper()
    # Specify our active learning model.
    df_train = fit_kwargs['train_dataset'].to_df()
    
    X_train, X_pool, y_train, y_pool = train_test_split(df_train['pairs'], df_train['labels'], 
                                                        train_size=N_INITIAL_TRAIN_SAMPLES, random_state=RS)
    
    print(f'Enter al_train X_train.shape[0]={X_train.shape[0]}, X_pool.shape[0]={X_pool.shape[0]}')
    
    learner = ActiveLearner(
      estimator=sk_ditto,       
      X_training=X_train.to_numpy(),
      y_training=y_train.to_numpy(),       
      query_strategy=preset_uncertainty_sampling, # preset_batch,
      **fit_kwargs
    )

    # Pool-based sampling    
    
    
    performance_history = []
    
    for index in range(N_AL_LABELING_QUERIES):
        query_index, query_instance = learner.query(X_pool.to_numpy(), **fit_kwargs)
        
        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool.iloc[query_index], y_pool.iloc[query_index]
        learner.teach(X=X.to_numpy(), y=y.to_numpy(), **fit_kwargs) # add to training data the newly queried X + labels
        
        # Remove the queried instance from the unlabeled pool.
        X_pool = X_pool.drop(X_pool.index[query_index])
        y_pool = y_pool.drop(y_pool.index[query_index])
                        
        # Calculate and report our model's accuracy.
        #model_accuracy = learner.score(None, None, **fit_kwargs)
        #print('f1 after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        
        # Save our model's performance for plotting.
        # performance_history.append(model_accuracy)