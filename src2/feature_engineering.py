from src2.config import K_B, MU_0, M_P, GAMMA

def engineer_features(df):
    df = df.copy()
    B_si = df['B_total'] * 1e-9
    n_si = df['proton_density'] * 1e6
    T_si = df['proton_thermal']
    V_si = df['proton_bulk_speed'] * 1e3
    p_thermal = n_si * K_B * T_si
    p_magnetic = B_si ** 2 / (2 * MU_0)
    df['plasma_beta'] = p_thermal / (p_magnetic + 1e-20)
    mass_density = n_si * M_P
    df['alfven_speed'] = (B_si / np.sqrt(MU_0 * mass_density + 1e-20)) / 1e3
    df['sound_speed'] = np.sqrt((GAMMA * K_B * T_si) / M_P) / 1e3
    df['sonic_mach_number'] = (V_si / 1e3) / (df['sound_speed'] + 1e-9)
    df['alfven_mach_number'] = (V_si / 1e3) / (df['alfven_speed'] + 1e-9)
    flux_cols = [f'flux_{i}' for i in range(50) if f'flux_{i}' in df.columns]
    if flux_cols:
        flux_data = df[flux_cols].values
        total_flux = flux_data.sum(axis=1, keepdims=True)
        p_i = (flux_data + 1e-20) / (total_flux + 1e-9)
        df['flux_entropy'] = -(p_i * np.log2(p_i)).sum(axis=1)
        df['entropy_gradient'] = df['flux_entropy'].diff().abs()
    for col in ['Bx_gsm', 'By_gsm', 'Bz_gsm', 'B_total']:
        if col in df.columns:
            df[f'd{col}_dt'] = df[col].diff()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
