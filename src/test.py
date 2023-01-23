from util import *

df = load_pickle("datasets/data.pkl")

df["l_director"] = df["directorIds"].apply(lambda x:len(x))
df["l_writer"] = df["writerIds"].apply(lambda x:len(x))
df["l_starring"] = df["starringIds"].apply(lambda x:len(x))
print(df["l_director"].describe())
print(df["l_writer"].describe())
print(df["l_starring"].describe())
print(df[(df["l_director"]<=2)&(df["l_writer"]<=4)&(df["l_starring"]<=6)])