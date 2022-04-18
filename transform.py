"""transform books and jester datasets into a format compatiple with Elyot"""

import numpy as np
import pandas as pd
from tqdm import tqdm

def read_in_books():
    wq = "./data/books/"
    di = {}
    du = {}
    df = pd.read_csv(wq + "BX-Book-Ratings.csv",encoding = 'unicode_escape', error_bad_lines = False, sep=";")

    df = df.sample(frac=0.1)
    df["Book-Rating"] = df["Book-Rating"].apply(lambda r : round(r/2, 1))
    df["Book-Rating"] = df["Book-Rating"].replace(to_replace = 0.0, value = 1.0)

    for i, u in enumerate(df["ISBN"].unique()):
        di[u] = i + 1
        
    for i, u in enumerate(df["User-ID"].unique()):
        du[u] = i + 1
        
    df["User-ID"] = df["User-ID"].apply(lambda v : int(du[v]))
    df["ISBN"] = df["ISBN"].apply(lambda v : int(di[v]))
    df = df.rename(columns = {"User-ID":"user_id", "ISBN" : "item_id", "Book-Rating" : "rating"})
    
    df_test = df.sample(frac=0.2)
    df_train = df[~df.isin(df_test)].dropna()
    df_train["user_id"] = df_train["user_id"].apply(lambda v : int(v))
    df_train["item_id"] = df_train  ["item_id"].apply(lambda v : int(v))

    df_train.to_csv(wq + "train_ratings.txt",sep = "\t", index=False, header=True)
    df_test.to_csv(wq + "test_ratings.txt",sep = "\t", index=False, header=True)


def read_in_jokes():
    wq = "./data/jester/"

    df_new = pd.DataFrame({"user_id" : [np.NaN], "item_id" : [np.NaN], "rating" : [np.NaN]})
    df = pd.read_excel(wq + "jester-data-2.xls")

    df = df.replace(99.00, np.NaN)
    df = df.apply(lambda r : round((r+10)/4, 1))
    for u_id, row in tqdm(df.iterrows()):
        try:
            for j_id, items in row.iteritems():
                if not np.isnan(items):
                    df_new.loc[len(df_new.index)] = [int(u_id), int(j_id.split("_")[1]), max(items, 1.0)]
        except KeyboardInterrupt:
            break
        
    df_new = df_new.sample(n=100000)
    df_test = df_new.sample(frac=0.2)
    df_train = df_new[~df_new.isin(df_test)].dropna()
    df_train["user_id"] = df_train["user_id"].apply(lambda v : int(v))
    df_train["item_id"] = df_train  ["item_id"].apply(lambda v : int(v))
 
    df_train.to_csv(wq + "train_ratings.txt",sep = "\t", index=False, header=True)
    df_test.to_csv(wq + "test_ratings.txt",sep = "\t", index=False, header=True)
    
    
if __name__ == "__main__":
    read_in_books()
    print("Done Books")
    read_in_jokes()
    print("Done Jokes")
