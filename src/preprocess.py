from movie import Movie, People, get_movie_data
from util import *

def train_val_test_split(df:pd.DataFrame,train_rate=0.8,val_rate=0.1,test_rate=0.1):
    df_sample = df.copy()
    df_sample = df_sample.sample(frac=1,random_state=42)

    # 念の為、正規化
    s = train_rate + val_rate + test_rate
    train_rate = train_rate / s
    val_rate = val_rate / s
    test_rate = test_rate / s

    val_amount = int(len(df) * val_rate)
    test_amount = int(len(df) * test_rate)

    val = df_sample[:val_amount]
    val = val.reset_index(drop=True)
    test = df_sample[val_amount:val_amount+test_amount]
    test = test.reset_index(drop=True)
    train = df_sample[val_amount+test_amount:]
    train = train.reset_index(drop=True)
    
    return train,val,test

if __name__ == "__main__":
    # movie_dict = get_movie_data()
    # df = movie2pd(movie_dict)
    # save_pickle("datasets/data.pkl",df)
    df = load_pickle("datasets/data.pkl")
    
    vocab = create_vocab(df,"datasets/vocab.tsv")

    df["l_director"] = df["directorIds"].apply(lambda x:len(x))
    df["l_writer"] = df["writerIds"].apply(lambda x:len(x))
    df["l_starring"] = df["starringIds"].apply(lambda x:len(x))

    # 監督2人, 脚本家4人, 主演6人までのデータに限定
    df = df[(df["l_director"]<=2)&(df["l_writer"]<=4)&(df["l_starring"]<=6)]
    df = df.drop(["l_director","l_writer","l_starring"],axis=1)

    for i in range(2):
        df["directorId"+str(i+1)] = [people[i].nameId if len(people) > i else "nm0000000" for people in df["directorIds"]]
    df = df.drop(["directorIds"],axis=1)

    for i in range(4):
        df["writerId"+str(i+1)] = [people[i].nameId if len(people) > i else "nm0000000" for people in df["writerIds"]]
    df = df.drop(["writerIds"],axis=1)

    for i in range(6):
        df["starringId"+str(i+1)] = [people[i].nameId if len(people) > i else "nm0000000" for people in df["starringIds"]]
    df = df.drop(["starringIds"],axis=1)

    train, val, test = train_val_test_split(df)

    save_pickle("datasets/train.pkl",train)
    save_pickle("datasets/val.pkl",val)
    save_pickle("datasets/test.pkl",test)

    