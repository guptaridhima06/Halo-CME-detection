# src1/data_loader.py

import pandas as pd
import numpy as np
import warnings

def prepare_data_streams(data_path, params):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = pd.read_csv(data_path, parse_dates=['time']).set_index('time')

    # STEPS
    steps_cols = [f'ep_inner_bin_{i}' for i in range(11)] + [f'ep_outer_bin_{i}' for i in range(11)]
    steps_data = df[steps_cols].copy().dropna(how='all')
    steps_data = steps_data.loc[steps_data['ep_inner_bin_0'].diff() != 0]
    steps_data['flux'] = steps_data[[f'ep_inner_bin_{i}' for i in range(11)]].sum(axis=1)
    high_cols = [f'ep_inner_bin_{i}' for i in range(params.high_energy_bin_start, 11)]
    steps_data['high_energy_flux'] = steps_data[high_cols].sum(axis=1)
    steps_data = steps_data[steps_data['flux'] > 1e-5].copy()

    # SWIS
    swis_cols = ['proton_density_blk', 'proton_thermal_blk', 'proton_bulk_speed_blk']
    swis_data = df[swis_cols].copy()
    for col in swis_cols:
        swis_data[col] = swis_data[col].mask(swis_data[col] == -1.0e31, np.nan)
    swis_data.dropna(inplace=True)

    return steps_data, swis_data
