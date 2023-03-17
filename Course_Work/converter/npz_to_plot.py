import pandas as pd
import os
import ecg_plot
import numpy as np
import sys


def npz_to_plot(df: pd.DataFrame,
                path_to_plot: str = './'):

    if not os.path.isdir(path_to_plot):
        os.makedirs(path_to_plot)

    full_path = path_to_plot + '/combined/'
    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    for i, loc in enumerate(df['fpath']):
        data = np.load(loc)['arr_0']
        ecg_plot.plot(ecg=data, sample_rate=500, title='', show_grid=True, columns=2)
        ecg_plot.save_as_png(file_name=df.loc[i, 'file_name'], path=full_path, show_grid=True, dpi=75)
        print("{}/{}".format(i + 1, df.shape[0]), end='\r')


if __name__ == "__main__":

    try:
        df = pd.read_csv(sys.argv[1])
        path_to_plot = sys.argv[2]
    except:
        raise Exception("Please input the following data in the given order:\n1) Path to the PTB-XL csv file,\n"
              "2) Path to the directory, to save the plots")

    npz_to_plot(df, path_to_plot)
