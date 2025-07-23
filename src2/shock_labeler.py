from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import numpy as np
import pandas as pd


def find_rh_confirmed_shocks(df):
    X = df[['proton_bulk_speed', 'proton_density', 'proton_thermal', 'B_total']].values
    mahal_dist = np.zeros(len(df))
    win = 120
    for i in range(win, len(X)):
        window = X[i - win:i]
        mean = np.mean(window, axis=0)
        try:
            inv_cov = np.linalg.inv(np.cov(window, rowvar=False))
            mahal_dist[i] = mahalanobis(X[i], mean, inv_cov)
        except:
            mahal_dist[i] = 0
    df['mahalanobis_dist'] = mahal_dist
    df['shock_candidate_flag'] = (mahal_dist > chi2.ppf(0.999, df=4)).astype(int)
    df['RH_confirmed_shock'] = 0
    df['compression_ratio'] = np.nan
    for idx in df[df['shock_candidate_flag'] == 1].index:
        up_start = idx - pd.Timedelta(minutes=5)
        up_end = idx - pd.Timedelta(seconds=5)
        down_start = idx + pd.Timedelta(seconds=5)
        down_end = idx + pd.Timedelta(minutes=5)
        if up_start in df.index and down_end in df.index:
            rho_up = df.loc[up_start:up_end, 'proton_density'].mean()
            rho_down = df.loc[down_start:down_end, 'proton_density'].mean()
            if rho_up > 0 and (rho_down / rho_up) > 1.5:
                df.loc[idx, 'RH_confirmed_shock'] = 1
                df.loc[idx, 'compression_ratio'] = rho_down / rho_up
    return df
