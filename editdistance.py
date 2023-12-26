import edlib
import pandas as pd
from tqdm import tqdm as tqdm_progress
import matplotlib.pyplot as plt
from align import *

# Create a tqdm wrapper for pandas read_csv
def tqdm_read_csv(file_path, delimiter='\t',chunksize=1000000,total_lines=30000000,**kwargs):
    
    with tqdm_progress(total=total_lines, desc='Loading') as pbar:
        for df_chunk in pd.read_csv(file_path, delimiter=delimiter, chunksize=chunksize, **kwargs):
            pbar.update(len(df_chunk))
            yield df_chunk


def data_loader(dataset="datasets\ERR240727_1_E2_30million.txt", total_lines=30000000,chunksize=1000000,n_chunk=3):

    print("starting data load...")



    # Use tqdm_read_csv to load the data with a progress bar
    #for chunk in tqdm_read_csv(dataset, names=['Column1', 'Column2']):
        # Process or manipulate the DataFrame chunk as needed
        #print(chunk)

    dfs = []
    # Use tqdm_read_csv to load the data with a progress bar

  
    with tqdm_progress(total=n_chunk, desc='Chunk Loading progress') as pbar:
        for chunk in tqdm_read_csv(dataset,chunksize=chunksize,total_lines=total_lines, names=['a', 'b']):
            # Apply the edit(1,2) function to calculate values for Column3

            chunk['dist'] = chunk.apply(lambda row: edlib.align(row['a'], row['b'])["editDistance"], axis=1)
            # Process or manipulate the DataFrame chunk as needed
            pbar.update(1)
            dfs.append(chunk)

            n_chunk=n_chunk-1

            if n_chunk==0:
                break

    # Merge the chunks if needed
    df = pd.concat(dfs, ignore_index=True)

    #check that the dist column is all integers

    #histogram of the edit distance, with dist>10 clumped together
    #df['dist'].hist()
    #I want hist for each individual dist
    #df['dist'].value_counts().plot(kind='bar')
    #plt.show()

    return df

def perform_align(df,D=4000,scale=0.1,N=None):
    bases,permvec=get_bases(D=D)
    df["align_sim"]=df.progress_apply(lambda row: get_align_sim(np.array([row['a']]),np.array([row['b']]),bases,permvec,scale,N=N), axis=1)
    return df

if __name__ == "__main__":

    total_lines=30000000
    chunksize=10000 #1000000
    n_chunk= total_lines//chunksize
    dataset="datasets\ERR240727_1_E2_30million.txt"
    df=data_loader(dataset,total_lines,chunksize,n_chunk)
    #check that the dist column is all integers
    print(df.dtypes)

    #histogram of the edit distance, with dist>10 clumped together
    #df['dist'].hist()
    #I want hist for each individual dist
    #df['dist'].value_counts().plot(kind='bar')

    tqdm_progress.pandas()

    D=4000

    for s in [0.01,0.1,1,10,100]:
        df_align=perform_align(df,scale=s,D=D,N=None)
        cols_to_keep=['dist','align_sim']
        df_align=df_align[cols_to_keep]
        data_name=dataset.split("\\")[-1].split(".")[0]
        df_align.to_csv(f"align_sim_{data_name}_{s}_{D}.csv",index=False)


    df=perform_align(df,scale=0.1,D=D,N=None)
    #Plot correlation between edit distance and align sim
    df.plot.scatter(x='dist', y='align_sim')
    plt.show()

    total_pair=df.shape[0]

    while True:
        
        T=float(input("Enter T (Threshold for Similarity): "))
        E=int(input("Enter E (Edit distance cutoff): "))

        pass_threshold_true=df[(df['align_sim']>=T) & (df['dist']<=E)].shape[0]
        pass_threshold_false=df[(df['align_sim']>=T) & (df['dist']>E)].shape[0]

        total_pass=pass_threshold_true+pass_threshold_false

        reject_threshold_true=df[(df['align_sim']<T) & (df['dist']>E)].shape[0]
        reject_threshold_false=df[(df['align_sim']<T) & (df['dist']<=E)].shape[0]


        print(f"Percentage of pairs that passed through filter: {(total_pass/total_pair)*100}")
        print(f"Percentage falsely passed through filter out of total_pair: {(pass_threshold_false/total_pair)}\n")

        print(f"Num falsely accepted: {pass_threshold_false}")
        print(f"falsely accept rate: {(pass_threshold_false/total_pass)*100}\n")





        

        total_reject=reject_threshold_true+reject_threshold_false

        print(f"Percentage of pairs that rejected through filter out of total_pair: {(total_reject/total_pair)*100}\n")
        print(f"Num falsely rejected: {reject_threshold_false}")
        print(f"Percentage falsely rejected through filter out of total_pair: {(reject_threshold_false/(total_pair))}")




