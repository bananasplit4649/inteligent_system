import argparse

from movie import Movie, People, get_movie_data
from util import *
from models import InteractionModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use GPU!") if torch.cuda.is_available() else print("Use CPU!")

model_path = "bert-base-cased"

def train_model(dataset_train:Dataset, dataset_val:Dataset,
                lr:float, batch_size:int, epoch_num:int, 
                drop_rate:float, patience:int,weight:list):
    model = InteractionModel(model_path,drop_rate=drop_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    print("Start training")
    for epoch in range(epoch_num):
        running_loss = 0
        model.train()
        data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        for data in tqdm(data_loader):
            ids_title = data["ids_title"].to(device)
            mask_title = data["mask_title"].to(device)
            directorId1 = data["directorId1"].to(device)
            directorId2 = data["directorId2"].to(device)

            writerId1 = data["writerId1"].to(device)
            writerId2 = data["writerId2"].to(device)
            writerId3 = data["writerId3"].to(device)
            writerId4 = data["writerId4"].to(device)

            starringId1 = data["starringId1"].to(device)
            starringId2 = data["starringId2"].to(device)
            starringId3 = data["starringId3"].to(device)
            starringId4 = data["starringId4"].to(device)
            starringId5 = data["starringId5"].to(device)
            starringId6 = data["starringId6"].to(device)

            rating = data['rating'].reshape(len(ids_title),1).to(device)

            optimizer.zero_grad()
            output = model(
                    ids_title,mask_title,
                    directorId1,directorId2,
                    writerId1,writerId2,writerId3,writerId4,
                    starringId1,starringId2,starringId3,starringId4,starringId5,starringId6
                            )

            loss = criterion(output, rating)

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
        
        model.eval()
        val_loss = 0
        data_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        for data in tqdm(data_loader):
            ids_title = data["ids_title"].to(device)
            mask_title = data["mask_title"].to(device)

            directorId1 = data["directorId1"].to(device)
            directorId2 = data["directorId2"].to(device)

            writerId1 = data["writerId1"].to(device)
            writerId2 = data["writerId2"].to(device)
            writerId3 = data["writerId3"].to(device)
            writerId4 = data["writerId4"].to(device)

            starringId1 = data["starringId1"].to(device)
            starringId2 = data["starringId2"].to(device)
            starringId3 = data["starringId3"].to(device)
            starringId4 = data["starringId4"].to(device)
            starringId5 = data["starringId5"].to(device)
            starringId6 = data["starringId6"].to(device)

            rating = data['rating'].reshape(len(ids_title),1).to(device)
            
            output = model(
                    ids_title,mask_title,
                    directorId1,directorId2,
                    writerId1,writerId2,writerId3,writerId4,
                    starringId1,starringId2,starringId3,starringId4,starringId5,starringId6
                            )

            loss = criterion(output, rating)
            
            val_loss += loss.data.item()

        early_stopping(
            epoch = epoch+1,
            lr = lr, 
            batch = batch_size,
            drop_rate = "{:.1e}".format(drop_rate),
            val_loss = val_loss,
            model = model,
            mode = args.mode)
            
        print("epoch:{}, train loss:{:0.4f}, validation loss:{:0.4f}".format(epoch+1, running_loss, val_loss))

        if early_stopping.early_stop:
            break


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--train_set_path", default="datasets/train.pkl", type=str, help="path to train set")
    parser.add_argument("-v","--validation_set_path", default="datasets/val.pkl", type=str, help="path to validation set")
    parser.add_argument("-b","--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("-lr","--learing_rate", default=2.0e-6, type=float, help="learning rate")
    parser.add_argument("--epoch", default=100, type=int, help="number of the epochs")
    parser.add_argument("--drop_rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--patience", default=5, type=int, help="patience")
    parser.add_argument("--vocab_path", default="datasets/vocab.tsv", type=str, help="path to vocabulary")
    parser.add_argument("--mode", default=0, type=int, help="0: rotten, 1:fresh")

    args = parser.parse_args()

    vocab = read_lines(args.vocab_path)
    vocab = [line.split("\t") for line in vocab]
    vocab = {line[0]:i for i,line in enumerate(vocab)}

    train,weight = create_dataset(args.train_set_path,vocab,args.mode)
    val,_ = create_dataset(args.validation_set_path,vocab,args.mode)

    print()
    print("Training the model using the following parameters")
    for k,v in args.__dict__.items():
        print(k,"\t",v)
    print()

    train_model(
        train,val,
        lr=args.learing_rate,
        batch_size=args.batch_size,
        epoch_num= args.epoch,
        drop_rate=args.drop_rate,
        patience=args.patience,
        weight=weight
        )

