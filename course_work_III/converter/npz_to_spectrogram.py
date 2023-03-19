import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib

matplotlib.use('Agg')

leads_to_names = {0: 'I',
                  1: 'II',
                  2: 'III',
                  3: 'aVR',
                  4: 'aVL',
                  5: 'aVF',
                  6: 'V1',
                  7: 'V2',
                  8: 'V3',
                  9: 'V4',
                  10: 'V5',
                  11: 'V6'}


def npz_to_spectrogram(df: pd.DataFrame,
                       path_to_spectrogram: str = './',
                       lead: int = 1):

    if not os.path.isdir(path_to_spectrogram):
        os.makedirs(path_to_spectrogram)

    full_path = path_to_spectrogram + "/{}_lead".format(leads_to_names[lead])
    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    fig = plt.figure(figsize=(3.31, 3.33))
    for i, loc in enumerate(df['fpath']):
        data = np.load(loc)['arr_0']
        plt.specgram(data[lead], cmap='jet')[3]
        plt.axis('off')
        plt.savefig("{}/{}".format(full_path, df.loc[i, 'file_name']), bbox_inches='tight', pad_inches=0)
        plt.clf()
        print("{}/{}".format(i + 1, df.shape[0]), end='\r')
    plt.close(fig)


if __name__ == "__main__":

    try:
        df = pd.read_csv(sys.argv[1])
        path_to_spectrogram = sys.argv[2]
        lead = int(sys.argv[3])
    except:
        raise Exception("Please input the following data in the given order:\n1) Path to the PTB-XL csv file,\n"
              "2) Path to the directory, to save the spectrograms,\n3) Lead, that you want to convert")

    npz_to_spectrogram(df, path_to_spectrogram, lead)
