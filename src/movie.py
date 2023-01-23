import pandas as pd
from tqdm import tqdm
from util import save_pickle

class Movie:
    def __init__(self,title:str,titleId:str,ordering:int):
        self.title = title
        self.titleId = titleId
        self.ordering = ordering
        self.directors = None
        self.writers = None
        self.starrings = None
        self.rating = None

    def set_directors(self,director:str):
        if self.directors is not None:
            self.directors.append(director)
        else:
            self.directors = [director]

    def set_writers(self,writer:str):
        if self.writers is not None:
            self.writers.append(writer)
        else:
            self.writers = [writer]
    
    def set_starrings(self,starring:str):
        if self.starrings is not None:
            self.starrings.append(starring)
        else:
            self.starrings = [starring]
    
    def set_rating(self,rating):
        self.rating = rating

    def del_less_popular(self,thresh):
        if self.directors is not None:
            self.directors = [people for people in self.directors if people.cnt > thresh]
            if len(self.directors) == 0:
                self.directors = None
        if self.writers is not None:  
            self.writers = [people for people in self.writers if people.cnt > thresh]
            if len(self.writers) == 0:
                self.writers = None
        if self.starrings is not None:
            self.starrings = [people for people in self.starrings if people.cnt > thresh]
            if len(self.starrings) == 0:
                self.starrings = None

    def all_applicable(self):
        return self.directors is not None and self.writers is not None and self.starrings is not None and self.rating is not None

class People:
    def __init__(self,name:str,nameId:str):
        self.name = name
        self.nameId = nameId
        self.cnt = 0

def get_movie_data(
    path_to_tile="datasets/title.akas.tsv",
    path_to_people="datasets/name.basics.tsv",
    path_to_principals="datasets/title.principals.tsv",
    path_to_ratings="datasets/title.ratings.tsv",
    thresh=4):
    print("Load Title")
    df = pd.read_table(path_to_tile,low_memory=False)
    
    # 対象言語は英語だけ
    df = df[df["language"]=="en"]

    print("Create Title Data")
    movie_dict = {}
    for index, row in tqdm(df.iterrows(),total=len(df)):
        if row["titleId"] in movie_dict and movie_dict[row["titleId"]].ordering < row["ordering"]:
            movie_dict[row["titleId"]] = Movie(title=row["title"],titleId=row["titleId"],ordering=row["ordering"])
        else:
            movie_dict[row["titleId"]] = Movie(title=row["title"],titleId=row["titleId"],ordering=row["ordering"])

    print("Load People")
    df = pd.read_table(path_to_people)
    print("Create People Data")
    people_dict = {}
    for index, row in tqdm(df.iterrows(),total=len(df)):
        people_dict[row["nconst"]] = People(name=row["primaryName"],nameId=row["nconst"])

    print("Load Principals")
    df = pd.read_table(path_to_principals)
    print("Creat Pricipals Data")
    for index, row in tqdm(df.iterrows(),total=len(df)):
        if row["tconst"] in movie_dict and row["nconst"] in people_dict:
            people_dict[row["nconst"]].cnt += 1
            if row["category"] == "director":
                movie_dict[row["tconst"]].set_directors(people_dict[row["nconst"]])
            elif row["category"] == "writer":
                movie_dict[row["tconst"]].set_writers(people_dict[row["nconst"]])
            elif row["category"] in ["actor","actress"]:
                movie_dict[row["tconst"]].set_starrings(people_dict[row["nconst"]])

    print("Load Rating")
    df = pd.read_table(path_to_ratings)
    print("Create Rating Data")
    for index, row in tqdm(df.iterrows(),total=len(df)):
        if row["tconst"] in movie_dict:
            movie_dict[row["tconst"]].set_rating(row["averageRating"])    

    for v in movie_dict.values():
        v.del_less_popular(thresh)

    movie_dict = {k:v for k, v in movie_dict.items() if v.all_applicable()==True}
    print("length of data:",len(movie_dict))
    return movie_dict

if __name__ == "__main__":
    movie_dict = get_movie_data()