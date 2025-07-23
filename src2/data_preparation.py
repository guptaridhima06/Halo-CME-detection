import pandas as pd
import numpy as np
from src2.config import NAN_DROP_THRESHOLD


def load_and_prepare_data(df, is_test_data=False):
    df.replace([-1e31, -9999, -9.999e+03, -1.000e+31], np.nan, inplace=True)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.dropna(subset=['time'], inplace=True)
        df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep='first')]
    nan_ratios = df.isnull().mean()
    high_nan_cols = nan_ratios[nan_ratios > NAN_DROP_THRESHOLD].index
    df.drop(columns=high_nan_cols, inplace=True)
    rename_map = {
        'Bt': 'B_total',
        'proton_density_blk': 'proton_density',
        'proton_bulk_speed_blk': 'proton_bulk_speed',
        'proton_thermal_blk': 'proton_thermal'
    }
    df.rename(columns=rename_map, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    flux_cols = sorted([col for col in df.columns if 'integrated_flux' in col])
    flux_rename_map = {old_name: f'flux_{i}' for i, old_name in enumerate(flux_cols[:50])}
    df.rename(columns=flux_rename_map, inplace=True)
    core = ['proton_density', 'proton_bulk_speed', 'proton_thermal', 'Bx_gsm', 'By_gsm', 'Bz_gsm', 'B_total']
    flux_std = [f'flux_{i}' for i in range(50)]
    df = df[[col for col in (core + flux_std) if col in df.columns]]
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
