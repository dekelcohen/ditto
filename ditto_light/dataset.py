import pandas as pd
import torch

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 hp,                 
                 pairs=None,
                 labels=None
                 ):
        self.tokenizer = get_tokenizer(hp.lm)
        print(f'type(tokenizer) == {type(self.tokenizer)} ')
        self.pairs = []
        self.labels = []
        self.max_len = hp.max_len
        self.size = getattr(hp, 'size', None)
        self.hp = hp        
        self.arabert_prep = None
        
        if pairs is not None: # pairs is a array like (list, Series, np.array) of tuples with pairs of strings. 
            self.pairs = pairs
            assert labels is not None, "pairs is not None --> labels cannot be None"
            self.pairs = pairs
            self.labels = labels
        else: # Read from a list or a file and parse
            if isinstance(path, list):
                lines = path
            else:
                lines = open(path)
    
            for line in lines:
                s1, s2, label = line.strip().split('\t')
                self.pairs.append((s1, s2))
                self.labels.append(int(label))

        self.pairs = self.pairs[:self.size]
        self.labels = self.labels[:self.size]
        self.da = getattr(hp, 'da', None)
        if self.da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None
            
    
    def to_df(self):
        return pd.DataFrame(data={ 'pairs' : self.pairs, 'labels': self.labels})
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        if self.hp.langid == 'ar':
            left = left.replace('COL name VAL','عمودي') # TODO: Fix for multiple columns 
            right = right.replace('COL name VAL','عمودي')
        
        # Arabertv02 preprocessor 
        if 'arabertv02' in self.hp.lm:
            from arabert.preprocess import ArabertPreprocessor
            if self.arabert_prep is None:            
                self.arabert_prep = ArabertPreprocessor(model_name=self.hp.lm)
            left = self.arabert_prep.preprocess(left)
            right = self.arabert_prep.preprocess(right)
            
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, x_aug, self.labels[idx]
        else:
            return x, self.labels[idx]


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)

