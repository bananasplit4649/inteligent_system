import torch
from transformers import BertModel

class SimpleModel(torch.nn.Module):
    def __init__(self,pretrained,input_size_title=768,name_vocab_size=45718,name_vec_size=64,drop_rate=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.name2vec = torch.nn.Linear(name_vocab_size,name_vec_size)
        self.fc =  torch.nn.Linear(input_size_title+name_vec_size*12,1)
        self.drop = torch.nn.Dropout(drop_rate)
        self.activate_func = torch.nn.ReLU()

    def forward(self,ids_title,mask_title,
                directorId1,directorId2,
                writerId1,writerId2,writerId3,writerId4,
                starringId1,starringId2,starringId3,starringId4,starringId5,starringId6):
        _, cls_title = self.bert(ids_title, attention_mask=mask_title)
        director1 = self.activate_func(self.name2vec(self.drop(directorId1)))
        director2 = self.activate_func(self.name2vec(self.drop(directorId2)))
        writer1 = self.activate_func(self.name2vec(self.drop(writerId1)))
        writer2 = self.activate_func(self.name2vec(self.drop(writerId2)))
        writer3 = self.activate_func(self.name2vec(self.drop(writerId3)))
        writer4 = self.activate_func(self.name2vec(self.drop(writerId4)))
        starring1 = self.activate_func(self.name2vec(self.drop(starringId1)))
        starring2 = self.activate_func(self.name2vec(self.drop(starringId2)))
        starring3 = self.activate_func(self.name2vec(self.drop(starringId3)))
        starring4 = self.activate_func(self.name2vec(self.drop(starringId4)))
        starring5 = self.activate_func(self.name2vec(self.drop(starringId5)))
        starring6 = self.activate_func(self.name2vec(self.drop(starringId6)))
        out = self.fc(
            torch.cat([
                cls_title,
                director1,director2,
                writer1,writer2,writer3,writer4,
                starring1,starring2,starring3,starring4,starring5,starring6], axis=1)
            )
        return out

class InteractionModel(torch.nn.Module):
    def __init__(self,pretrained,input_size_title=768,name_vocab_size=45718,name_vec_size=64,drop_rate=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.name2vec = torch.nn.Linear(name_vocab_size,name_vec_size)
        self.interaction = torch.nn.Linear(name_vec_size*12, name_vec_size*6)
        self.fc1 =  torch.nn.Linear(input_size_title+name_vec_size*6,1)
        self.drop = torch.nn.Dropout(drop_rate)
        self.activate_func = torch.nn.ReLU()

    def forward(self,ids_title,mask_title,
                directorId1,directorId2,
                writerId1,writerId2,writerId3,writerId4,
                starringId1,starringId2,starringId3,starringId4,starringId5,starringId6):
        _, cls_title = self.bert(ids_title, attention_mask=mask_title)
        director1 = self.activate_func(self.name2vec(self.drop(directorId1)))
        director2 = self.activate_func(self.name2vec(self.drop(directorId2)))
        writer1 = self.activate_func(self.name2vec(self.drop(writerId1)))
        writer2 = self.activate_func(self.name2vec(self.drop(writerId2)))
        writer3 = self.activate_func(self.name2vec(self.drop(writerId3)))
        writer4 = self.activate_func(self.name2vec(self.drop(writerId4)))
        starring1 = self.activate_func(self.name2vec(self.drop(starringId1)))
        starring2 = self.activate_func(self.name2vec(self.drop(starringId2)))
        starring3 = self.activate_func(self.name2vec(self.drop(starringId3)))
        starring4 = self.activate_func(self.name2vec(self.drop(starringId4)))
        starring5 = self.activate_func(self.name2vec(self.drop(starringId5)))
        starring6 = self.activate_func(self.name2vec(self.drop(starringId6)))
        out1 = self.interaction(
            self.drop(
            torch.cat([
                director1,director2,
                writer1,writer2,writer3,writer4,
                starring1,starring2,starring3,starring4,starring5,starring6], axis=1)
            ))
        h1 = self.activate_func(out1)
        out = self.fc1(
            self.drop(
            torch.cat([
                cls_title,h1], axis=1)
            ))
        return out