from glob import glob 
import pandas as pd
import numpy as np


files = glob("D:/three years data/*/*outcomes.csv")
df = pd.concat(map(pd.read_csv, files))

files_2 = glob("D:/three years data/*/*street.csv")
df2 = pd.concat(map(pd.read_csv, files_2))


def merge_clean(df1, df2, key):
    
    df_merged = pd.merge(df1, df2, how="inner", on = key)
    
    for col in df_merged.columns:
        if col.endswith("_x"):
            df_merged.rename(columns = lambda col:col.rstrip("_x"), inplace = True)
        elif col.endswith("_y"):
            to_drop = [col for col in df_merged if col.endswith("_y")]
            df_merged.drop(to_drop, axis = 1, inplace = True)
        else:
            pass
    return df_merged

df_final = merge_clean(df, df2, "Crime ID")
df_final = df_final.drop(['Reported by', 'Falls within', 'Context', 'Last outcome category','LSOA name','Location'], axis=1)
df_final = df_final.dropna()
df_final = df_final.reset_index(drop = True)

df_final = df_final.drop_duplicates(subset="Crime ID")
df_final = df_final.reset_index(drop = True)

df_final.rename(inplace=True, columns={
    'Crime ID': 'crime_id',
    'Month': 'date',
    'LSOA code': 'LSOA_code',
    'Crime type': 'crime_type',
    'Outcome type': 'outcome_type'})

df = df_final
del(df_final)
import gc 
gc.collect()

df["date"] = pd.to_datetime(df["date"])

df["crime_type"] = df["crime_type"].replace({"Other theft":"Theft", "Shoplifting":"Theft", 
"Bicycle theft": "Theft", "Theft from the person": "Theft"}, regex=True)

df["outcome_type"] = df["outcome_type"].replace({"Suspect charged":"prosecuted", 
"Suspect charged as part of another case":"prosecuted", 
"Defendant sent to Crown Court": "prosecuted", "Action to be taken by another organisation": "prosecuted",
"Offender given a drugs possession warning" : "prosecuted", "Offender given penalty notice":"prosecuted",
"prosecuted as part of another case":"prosecuted",
"Offender given a caution":"prosecuted", "prosecuted as part of another case":"prosecuted"})

df["outcome_type"] = df["outcome_type"].replace({"Unable to prosecute suspect":"not_prosecuted", 
                                                 "Local resolution":"not_prosecuted",
                                                 "Formal action is not in the public interest": "not_prosecuted",
                                                 "Further action is not in the public interest" : "not_prosecuted"})

df.drop(df.loc[df['outcome_type']=="Investigation complete; no suspect identified"].index, inplace=True)
df.drop(df.loc[df['outcome_type']=="Further investigation is not in the public interest"].index, inplace=True)

df.drop(["crime_id"], axis=1, inplace=True)


# Haversine Formula 
from math import radians, cos, sin, asin, sqrt

def haversine_dist(lon1, lat1, lon2, lat2):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)

    # Haversine formula 
    dlat = lat2 - lat1
    dlon = lon2 - lon1 
    a = np.sin(dlat / 2.0)** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371 # radius of earth in km

    distance = c * r

    return distance

uni_lonlat = -0.2396, 51.7519
devailland_lonlat = -0.2713, 51.7107
warnerbros_lonlat = -0.4172, 51.6930

def add_landmark_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_crime_distance'] = haversine_dist(lon, lat, df['Longitude'], df['Latitude'])


for a_df in [df]:
    for name, lonlat in [('university', uni_lonlat), ('museum', devailland_lonlat), ('studios', warnerbros_lonlat)]:
        add_landmark_distance(a_df, name, lonlat)

df.drop(["Longitude","Latitude"], axis = 1, inplace = True)

def year_month_extract(df, col):
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month

    return year_month_extract

year_month_extract(df, "date")

df.drop(["date"], axis=1, inplace=True)
df.reset_index(inplace=True)
df.drop(["index"], axis = 1, inplace=True)


dummy_year = pd.get_dummies(df["date_year"], prefix="crime_year")
dummy_month = pd.get_dummies(df["date_month"], prefix="crime_month")
dummy_crime_type = pd.get_dummies(df["crime_type"], prefix="crime_type")

df = pd.concat([df, dummy_year, dummy_month, dummy_crime_type], axis=1)
df.drop(["date_year", "date_month","crime_type"], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
lb_encode = LabelEncoder()
df['outcome_type'] = lb_encode.fit_transform(df['outcome_type'])

df_outcome = df["outcome_type"]
df = pd.concat([df, df_outcome], axis=1)

df = df.loc[:,~df.T.duplicated(keep='last')]
x_input = df.iloc[: , :-1]
y = df["outcome_type"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_input, y, test_size=0.20, random_state=4, stratify=y)

df_train = pd.concat([x_train, y_train], axis=1)
df_test = pd.concat([x_test, y_test], axis=1)

mean_encoded = df_train.groupby(['LSOA_code'])['outcome_type'].mean().to_dict()
df_train['LSOA_code'] =  df_train['LSOA_code'].map(mean_encoded)
df_test['LSOA_code'] =  df_test['LSOA_code'].map(mean_encoded)
df_test['LSOA_code'].fillna(df_test["LSOA_code"].mode()[0], inplace=True)


df_train.drop(["crime_year_2022","crime_month_12","crime_type_Other crime"], axis = 1, inplace = True)
df_test.drop(["crime_year_2022","crime_month_12","crime_type_Other crime"], axis = 1, inplace = True)


