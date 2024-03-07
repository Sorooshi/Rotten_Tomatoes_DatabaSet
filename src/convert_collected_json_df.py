import json
import numpy as np
import pandas as pd

EXTRACT_LARGE = True

FEATURES_1 = [
    'Title', 'Synopsis', 'Original Language', 'Runtime', 
    'Director', 'Producer', 'Writer', 'Top Cast',
    'Distributor', 'Production Co', 'Box Office (Gross USA)', 
    'Tomato Meter', 'Audience Score', 'No. Reviews', 'Genre', 
    'Release Date (Theaters)', 'Release Date (Streaming)', 'Link'
]

FEATURES_2 = [
        'Title', 'Synopsis', 'Original Language', 'Runtime', 
        'Director', 'Top Cast', 'Tomato Meter', 'Audience Score',
        'No. Reviews', 'Genre', 'Link']

def append_row(df, row):
    return pd.concat([
        df, pd.DataFrame([row], columns=row.index)
    ]).reset_index(drop=True)


def load_collected_json(path):
    with open(path,  'r') as fp:
        data = json.load(fp)
    return data


def get_medium_movies_df(json_data):
    links_with_issue = []
    errors = []
    medium_movies_data = pd.DataFrame(columns=FEATURES_1)

    for k, v in json_data.items():
        try:
            run_time =  int(v['Info']['Runtime'].split()[0].split("h")[0]) * 60 + \
                int(v['Info']['Runtime'].split()[1].split("m")[0])
            tmp_box_office = v['Info']['Box Office (Gross USA)'].strip().split("$")[1]
            
            if "M" in tmp_box_office:
                box_office = float(tmp_box_office.split("M")[0]) * 1000000
            elif "K" in tmp_box_office:
                box_office = float(tmp_box_office.split("K")[0]) * 1000
            else:
                box_office = 0.
            a_row = pd.Series({
                'Title': v['Title'].strip(),
                'Synopsis': v['Synopsis'], 
                'Original Language': v['Info']['Original Language'].strip(), 
                'Runtime': run_time,
                'Director': v['Info']['Director'].strip(), 
                'Producer': v['Info']['Producer'].strip(), 
                'Writer':  v['Info']['Writer'].strip(),
                'Top Cast': v["Top Cast"], 
                'Distributor': v['Info']['Distributor'].strip(),
                'Production Co': v['Info']['Production Co'].strip(),
                'Box Office (Gross USA)': box_office, 
                'Tomato Meter': float(v["Score Panel"][2].strip("%"))/100,
                'Audience Score': float(v["Score Panel"][5].strip("%"))/100,
                'No. Reviews': int(v["Score Panel"][4].split(" ")[0]),
                'All Genres': v['Info']['Genre'].strip(), 
                'Genre': v['Info']['Genre'].strip().split(", ")[0],
                'Release Date (Theaters)': v['Info']['Release Date (Theaters)'].strip(),
                'Release Date (Streaming)': v['Info']['Release Date (Streaming)'].strip(), 
                'Link': k.strip()
            })
            medium_movies_data = append_row(df=medium_movies_data, row=a_row)

        except Exception as error:
            print(
                f"In {k} \n"
                f"{error} \n"
                f"occurred !"
            )
            links_with_issue.append(k)
            errors.append(error)

    languages = list(medium_movies_data["Original Language"].unique())
    for language in languages:
        medium_movies_data['Original Language'].replace(language, language[:3], inplace=True)

    medium_movies_data.to_csv("./data/medium_movies_data.csv", index=False)

    return medium_movies_data

def get_large_movies_df(json_data):
    links_with_issue = []
    errors = []

    large_movies_data = pd.DataFrame(columns=FEATURES_2)

    for k, v in json_data.items():
        try:
            run_time =  int(v['Info']['Runtime'].split()[0].split("h")[0]) * 60 + \
                  int(v['Info']['Runtime'].split()[1].split("m")[0])
            a_row = pd.Series({
                'Title': v['Title'].strip(),
                'Synopsis': v['Synopsis'].strip(), 
                'Original Language': v['Info']['Original Language'].strip(), 
                'Runtime': run_time,
                'Director': v['Info']['Director'].strip(), 
                'Top Cast': v["Top Cast"], 
                'Tomato Meter': float(v["Score Panel"][2].strip("%"))/100,
                'Audience Score': float(v["Score Panel"][5].strip("%"))/100,
                'No. Reviews': int(v["Score Panel"][4].split(" ")[0]),
                'All Genres': v['Info']['Genre'].strip(), 
                'Genre': v['Info']['Genre'].strip().split(", ")[0],
                'Link': k.strip()
            })
            large_movies_data = append_row(df=large_movies_data, row=a_row)
        except Exception as error:
            print(
                f"In {k} \n"
                f"{error} \n"
                f"occurred !"
            )
            links_with_issue.append(k)
            errors.append(error)

        languages = list(large_movies_data["Original Language"].unique())
        for language in languages:
            large_movies_data['Original Language'].replace(language, language[:3], inplace=True)

        large_movies_data.to_csv("./data/large_movies_data.csv", index=False)

        return large_movies_data
                
    
if __name__ == "__main__":

    collected_json_data = load_collected_json(
        path="./data/rotten_tomatoes_movies_data_with_score_panels.json"
        )
    print(
        f" size of collected data: {len(collected_json_data)}"
        )
    
    if EXTRACT_LARGE is False:
        medium_movies_data = get_medium_movies_df(
            json_data=collected_json_data
            )
        
        print(medium_movies_data.head())

        print(
            f"medium df data {medium_movies_data.shape}"
        )
    
    if EXTRACT_LARGE:
        large_movies_data = get_large_movies_df(
            json_data=collected_json_data
            )
        
        print(large_movies_data.head())

        print(
            f"medium df data {large_movies_data.shape}"
        )
        
    
