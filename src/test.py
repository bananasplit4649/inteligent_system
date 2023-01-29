import argparse

from movie import Movie, People, get_movie_data
from util import *
from models import InteractionModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use GPU!") if torch.cuda.is_available() else print("Use CPU!")

model_path = "bert-base-cased"

def test_model(dataset_test:Dataset, batch_size:int):
    model = InteractionModel(model_path).to(device)
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    model.eval()
    print("Start inference")
    preds = None
    trues = None
    data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
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

        rating = data['rating'].reshape(len(ids_title),1)
        
        output = model(
                ids_title,mask_title,
                directorId1,directorId2,
                writerId1,writerId2,writerId3,writerId4,
                starringId1,starringId2,starringId3,starringId4,starringId5,starringId6
        )

        if preds is None and trues is None:
            preds = output.to('cpu').detach().numpy()
            trues = rating.detach().numpy()
        else:
            preds = np.concatenate([preds, output.to('cpu').detach().numpy()])
            trues = np.concatenate([trues, rating.detach().numpy()])
    preds = [1 if pred > 0.5 else 0 for pred in preds]
    trues = [int(t) for t in trues]
    print(classification_report(trues,preds))
    print(confusion_matrix(trues,preds))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test_set_path", default="datasets/test.pkl", type=str, help="path to test set")
    parser.add_argument("-b","--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("-m","--model_path", default="model/InteractionModel.lr2e-06.batch8.drop1.0e-01.epoch1.pt", type=str, help="path to model")
    parser.add_argument("--vocab_path", default="datasets/vocab.tsv", type=str, help="path to vocabulary")
    parser.add_argument("--mode", default=0, type=int, help="0: rotten, 1:fresh")

    args = parser.parse_args()

    vocab = read_lines(args.vocab_path)
    vocab = [line.split("\t") for line in vocab]
    vocab = {line[0]:i for i,line in enumerate(vocab)}

    test,_ = create_dataset(args.test_set_path,vocab,args.mode)

    test_model(
        test,
        batch_size=args.batch_size,
        )