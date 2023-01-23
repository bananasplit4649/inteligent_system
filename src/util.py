import pickle
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore', FutureWarning)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer

def save_pickle(path,save_object):
    with open(path,"wb") as f:
        pickle.dump(save_object,f)

def load_pickle(path):
    with open(path,"rb") as f:
        load_object = pickle.load(f)
    return load_object

def write_lines(path,lines):
    lines = [line.strip()+"\n" for line in lines]
    with open(path,"w") as f:
        f.writelines(lines)

def read_lines(path):
    with open(path,"r") as f:
        lines = f.readlines()
    return lines

def movie2pd(movie_dict:dict):
    df = pd.DataFrame([], columns=["title","directorIds","writerIds","starringIds","rating"])
    for v in movie_dict.values():
        title = v.title
        directorIds = v.directors
        writerIds = v.writers
        starringIds = v.starrings
        rating = v.rating
        addRow = pd.Series([title,directorIds,writerIds,starringIds,rating], index=df.columns)
        df = df.append(addRow, ignore_index=True)
    return df

def create_vocab(df:pd.DataFrame,save_path:str):
    vocab = set()
    people_lists = df["directorIds"].tolist() + df["writerIds"].tolist() + df["starringIds"].tolist()
    for people_list in people_lists:
        for people in people_list:
            vocab.add(people.nameId+"\t"+people.name)
    vocab = list(vocab)
    vocab = sorted(vocab)
    vocab = ["nm0000000\tN/A"] + vocab
    write_lines(save_path,vocab)
    return vocab

def nameIds2onehot(vocab:list,nameId:str):
    nameIdVocab = [tmp.split("\t")[0] for tmp in vocab]
    nameIdOnehot = np.eye(len(nameIdVocab))[[nameIdVocab.index(nameId)]][0]
    return nameIdOnehot

class CreateDataset(Dataset):
    def __init__(self, title:str, directorIds:pd.DataFrame, writerIds:pd.DataFrame, starringIds:pd.DataFrame, rating:float, vocab:list, tokenizer, max_len:int):
        self.title = title
        self.directorIds = directorIds
        self.writerIds = writerIds
        self.starringIds = starringIds
        self.rating = rating
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.rating)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        title = self.title[index]
        inputs = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
        )
        ids_title = inputs['input_ids']
        mask_title = inputs['attention_mask']

        idx_directors = [None for _ in range(2)]
        for i in range(2):
            idx_directors[i] = nameIds2onehot(self.vocab,self.directorIds["directorId"+str(i+1)][index])

        idx_writers = [None for _ in range(4)]
        for i in range(4):
            idx_writers[i] = nameIds2onehot(self.vocab,self.writerIds["writerId"+str(i+1)][index])

        idx_starrings = [None for _ in range(6)]
        for i in range(6):
            idx_starrings[i] = nameIds2onehot(self.vocab,self.starringIds["starringId"+str(i+1)][index])

        rating = self.rating[index]

        return {
            'ids_title': torch.LongTensor(ids_title),
            'mask_title': torch.LongTensor(mask_title),
            'directorId1':torch.FloatTensor(idx_directors[0]),
            'directorId2':torch.FloatTensor(idx_directors[1]),
            'writerId1':torch.FloatTensor(idx_writers[0]),
            'writerId2':torch.FloatTensor(idx_writers[1]),
            'writerId3':torch.FloatTensor(idx_writers[2]),
            'writerId4':torch.FloatTensor(idx_writers[3]),
            'starringId1':torch.FloatTensor(idx_starrings[0]),
            'starringId2':torch.FloatTensor(idx_starrings[1]),
            'starringId3':torch.FloatTensor(idx_starrings[2]),
            'starringId4':torch.FloatTensor(idx_starrings[3]),
            'starringId5':torch.FloatTensor(idx_starrings[4]),
            'starringId6':torch.FloatTensor(idx_starrings[5]),
            'rating': torch.from_numpy(np.asarray(rating)).float()
        }

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, epoch, lr, batch, drop_rate, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, lr, batch, drop_rate, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, lr, batch, drop_rate, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, epoch, lr, batch, drop_rate, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if drop_rate is None:
            torch.save({"epoch": epoch,"model_state_dict":model.state_dict()}, f"model/{model.__class__.__name__}.lr{lr}.batch{batch}.epoch{epoch}.undersampling.pt")
        else:
            torch.save({"epoch": epoch,"model_state_dict":model.state_dict()}, f"model/{model.__class__.__name__}.lr{lr}.batch{batch}.drop{drop_rate}.epoch{epoch}.pt")
        self.val_loss_min = val_loss

def create_dataset(path:str,vocab:list):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    df = load_pickle(path)
    dataset = CreateDataset(
                df["title"], df[["directorId1","directorId2"]], \
                df[["writerId1","writerId2","writerId3","writerId4"]],\
                df[["starringId1","starringId2","starringId3","starringId4","starringId5","starringId6"]],\
                df["rating"], vocab,\
                tokenizer, 128
                )
    return dataset