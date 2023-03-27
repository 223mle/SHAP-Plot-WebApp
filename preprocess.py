import pandas as pd
import numpy as np
import json
from functools import partial
from collections import defaultdict, Counter


class Preprocess:
    def __init__(self, credit_df, movie_df):
        self.credit = credit_df
        self.movie = movie_df

    def preprocess(self):
        self.credit.columns = ['id', 'title', 'cast', 'crew']
        df = self.movie.merge(self.credit[['id', 'cast', 'crew']], on='id')

        # to datetime
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        # json形式で格納されているcolumnsに対して前処理
        json_columns = {'cast', 'crew', 'genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages'}

        for c in json_columns:
            df[c] = df[c].apply(json.loads)
            if c != "crew":
                df[c] = df[c].apply(lambda row: [x["name"] for x in row])

        # create director writer and producer columns
        def get_job(job, row):
            person_name = [x['name'] for x in row if x['job']==job]
            return person_name[0] if len(person_name) else np.nan

        df["director"] = df["crew"].apply(partial(get_job, "Director"))
        df["writer"]   = df["crew"].apply(partial(get_job, "Writer"))
        df["producer"] = df["crew"].apply(partial(get_job, "Producer"))

        # create profit column
        df["profit"] = df["revenue"] - df["budget"]

        # 最頻値を用いて欠損値埋め
        for col in ['runtime', 'release_year', 'release_month']:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

        return df
