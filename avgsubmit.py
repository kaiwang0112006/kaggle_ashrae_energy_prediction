# -*-coding: utf-8-*-
#import sys
#sys.path.append('/data/tinyv/kw/ppackage')
import pandas as pd
from project_demo.tools.evaluate import *
from project_demo.tools.optimize import *
from project_demo.tools.multi_apply import *

def cal_mean(row,files):
    nums = []
    for f in files:
        nums.append(row[f])
    return np.mean(nums)


def main():
    files = ['submit_int.csv', 'submit_m2.csv']
    dl = []
    for f in files:
        dfp = pd.read_csv(f).rename(columns={'meter_reading': f})[[f]]
        dl.append(dfp)
    df = pd.concat(dl, axis=1).reset_index().rename(columns={'index': 'row_id'})
    df = apply_row_by_multiprocessing(df,cal_mean,files=files,workers=4)
    #df['meter_reading'] = df.apply(lambda row: cal_mean(row, files), axis=1)
    df[['row_id','meter_reading']].to_csv('submit_avg.csv', index=False)

if __name__ == '__main__':
    main()