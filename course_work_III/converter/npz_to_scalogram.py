import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
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


def npz_to_scalogram(df: pd.DataFrame,
                     path_to_scalogram: str = './',
                     lead: int = 1):

    if not os.path.isdir(path_to_scalogram):
        os.makedirs(path_to_scalogram)

    full_path = path_to_scalogram + "/{}_lead".format(leads_to_names[lead])
    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    widths = np.arange(1, 500)
    fig = plt.figure(figsize=(3.31, 3.33))
    for i, loc in enumerate(df['fpath']):
        data = np.load(loc)['arr_0']
        cwtmatr = signal.cwt(data[lead], signal.ricker, widths)
        plt.imshow(abs(cwtmatr), cmap='jet', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).min())
        plt.axis('off')
        plt.savefig("{}/{}".format(full_path, df.loc[i, 'file_name']), bbox_inches='tight', pad_inches=0)
        print("{}/{}".format(i + 1, df.shape[0]), end='\r')
        plt.clf()
    plt.close(fig)


if __name__ == "__main__":

    try:
        df = pd.read_csv(sys.argv[1])
        path_to_scalogram = sys.argv[2]
        lead = int(sys.argv[3])
    except:
        raise Exception("Please input the following data in the given order:\n1) Path to the PTB-XL csv file,\n"
              "2) Path to the directory, to save the scalograms,\n3) Lead, that you want to convert")

    npz_to_scalogram(df, path_to_scalogram, lead)
