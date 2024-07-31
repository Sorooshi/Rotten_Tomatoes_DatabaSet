import pickle
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="Convert JSON to DF")

parser.add_argument(
        "-s", "--size", default=0, type=int,
        help="To extract large-size data when it is set to 1, and medium-size else."
        )

def get_isolated_nodes(df):

    zero_rows = df.sum(axis=1) == 0
    zero_rows = zero_rows.loc[zero_rows == True].index
    zero_cols = df.sum(axis=0) == 0
    zero_cols = zero_cols.loc[zero_cols == True].index

    return zero_rows, zero_cols

def get_list_of_casts(x):
    x = x.split(", ")
    xx = []
    for k in range(len(x)):
        if k == 0:
            xx.append(x[k].split("[")[1])
        elif k % 2 == 0 and k != 0:
            xx.append(x[k])
        # elif k == len(x):
        #     xx.append(x.split("]")[0])
    return set(xx)

def get_list_of_others(x):
    return set(x.split(", "))

def get_edge_weight(a, b):
    return ((len(a.intersection(b)) / len(a)) + len(b.intersection(a)) / len(b)) / 2

def get_medium_adjacency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_np = df.values
    adjacency = np.zeros(shape=(len(df_np), len(df_np)))

    for i in range(len(df_np)):
        for j in range(i, len(df_np)):
            if i != j:
                # Directors
                weight_dir = get_edge_weight(
                a = get_list_of_others(df_np[i, 4]), 
                b = get_list_of_others(df_np[j, 4])
            )
                # Producers
                weight_pro = get_edge_weight(
                    a = get_list_of_others(df_np[i, 5]), 
                    b = get_list_of_others(df_np[j, 5])
                )
                # Writers
                weight_wri = get_edge_weight(
                    a = get_list_of_others(df_np[i, 6]), 
                    b = get_list_of_others(df_np[j, 6])
                )
                # Top casts
                weight_casts = get_edge_weight(
                    a = get_list_of_casts(df_np[i, 7]), 
                    b = get_list_of_casts(df_np[j, 7])
                )
                weight = weight_dir + weight_pro + weight_wri + weight_casts
            else:
                weight = 0
            
            adjacency[i, j] = weight
            adjacency[j, i] = weight


    data_df_a = pd.DataFrame(
        data=adjacency, 
        columns=df.Title.values,
        index=df.Title.values,
        )
    
    no_link_movies = list(data_df_a.loc[data_df_a.sum(axis=0) == 0].index)
    zero_rows, zero_cols = get_isolated_nodes(df=data_df_a)
    
    print(
        f"zero_rows: {zero_rows} \n", 
        f"zero_cols: {zero_cols} \n"
        f"size of zero_rows: {len(zero_rows)} " 
        f"size of zero_cols: {len(zero_rows)} " 
        f"size of no_link movies {len(no_link_movies)}"
    )

    data_df_a.drop(index=zero_rows, columns=zero_cols, inplace=True)


    data_a = pd.DataFrame(
        data=adjacency, 
        columns=None,
        )
    zero_rows, zero_cols = get_isolated_nodes(df=data_a)
    data_a.drop(index=zero_rows, columns=zero_cols, inplace=True)
    print(
        f"cleaned data shapes: {data_df_a.shape, data_a.shape}"
    )

    data_df_a.to_csv("./data/medium_data_df_a.csv", index=True)
    data_a.to_csv("./data/medium_data_a.csv", header=False, index=False)

    with open ("./data/medium_data_no_link_movies.pickle", "wb") as fp:
        pickle.dump(no_link_movies, fp)
    
    print(
        f"cleaned data shapes: {data_df_a.shape, data_a.shape}"
    )
        
    return data_df_a, data_a


def get_large_adjacency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_np = df.values
    adjacency = np.zeros(shape=(len(df_np), len(df_np)))

    for i in range(len(df_np)):
        for j in range(i, len(df_np)):
            if i != j:
                # Directors
                weight_dir = get_edge_weight(
                a = get_list_of_others(df_np[i, 4]), 
                b = get_list_of_others(df_np[j, 4])
                )
                # Top casts
                weight_casts = get_edge_weight(
                    a = get_list_of_casts(df_np[i, 5]), 
                    b = get_list_of_casts(df_np[j, 5])
                )
                weight = weight_dir + weight_casts
            else:
                weight = 0.
            
            adjacency[i, j] = weight
            adjacency[j, i] = weight


    data_df_a = pd.DataFrame(
        data=adjacency, 
        columns=df.Title.values,
        index=df.Title.values,
        )
    
    no_link_movies = list(data_df_a.loc[data_df_a.sum(axis=0) == 0].index)
    zero_rows, zero_cols = get_isolated_nodes(df=data_df_a)

    print(
        f"zero_rows: {zero_rows} \n", 
        f"zero_cols: {zero_cols} \n"
        f"size of zero_rows: {len(zero_rows)} " 
        f"size of zero_cols: {len(zero_rows)} " 
        f"size of no_link movies {len(no_link_movies)} "
        )
    
    data_df_a.drop(index=zero_rows, columns=zero_cols, inplace=True)
    print(data_df_a.shape)

    data_a = pd.DataFrame(
        data=adjacency, 
        columns=None,
        )
    zero_rows, zero_cols = get_isolated_nodes(df=data_a)
    data_a.drop(index=zero_rows, columns=zero_cols, inplace=True)
    print(data_a.shape)
    
    data_df_a.to_csv("./data/large_data_df_a.csv", index=True)
    data_a.to_csv("./data/large_data_a.csv", header=False, index=False)

    with open ("./data/large_data_no_link_movies.pickle", "wb") as fp:
        pickle.dump(no_link_movies, fp)
    
    return data_df_a, data_a



if __name__ == "__main__":

    args = parser.parse_args()
    extract_large = args.size

    if extract_large != 1:
        medium_df = pd.read_csv('./data/medium_movies_data.csv')
        print(
            f"Medium-size data head: {medium_df.head()} \n"
            f"Medium-size data shape: {medium_df.shape} \n", 
            )

        data_df_a, data_a = get_medium_adjacency_matrix(df=medium_df)
        
    elif extract_large == 1:
        large_df = pd.read_csv('./data/large_movies_data.csv')
        print(
            f"Medium-size data head: {large_df.head()} \n"
            f"Medium-size data shape: {large_df.shape} \n", 
            )

        data_df_a, data_a = get_large_adjacency_matrix(df=large_df)

