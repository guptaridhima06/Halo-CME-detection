# src1/shock_detection_stats.py

import pandas as pd
import numpy as np

class ShockWarningDetector:
    def __init__(self, steps_data, parameters):
        self.steps_data = steps_data
        self.params = parameters
        self.warnings = []
        self.flux_anomaly_history = []
        self.max_flux_anomaly_seen = 0.0

    def run_detection(self):
        baseline_steps = int(self.params.particle_baseline_hours * 6)
        active_persistence_counter = 0
        cooldown_until_index = -1

        for i in range(baseline_steps, len(self.steps_data)):
            if i <= cooldown_until_index:
                self.flux_anomaly_history.append((self.steps_data.index[i], 1.0))
                continue

            baseline_window = self.steps_data.iloc[i - baseline_steps:i]
            quiet_flux_baseline = baseline_window['flux'].median()
            quiet_flux_std = baseline_window['flux'].std()
            quiet_flux_iqr = baseline_window['flux'].quantile(0.75) - baseline_window['flux'].quantile(0.25)

            dynamic_threshold = quiet_flux_baseline + 1.5 * quiet_flux_iqr
            current_flux = self.steps_data['flux'].iloc[i]
            anomaly_score = current_flux / (quiet_flux_baseline + 1e-9)

            self.flux_anomaly_history.append((self.steps_data.index[i], anomaly_score))
            self.max_flux_anomaly_seen = max(self.max_flux_anomaly_seen, anomaly_score)

            if current_flux >= dynamic_threshold:
                active_persistence_counter += 1
            else:
                active_persistence_counter = 0

            dynamic_persistence = 2 if quiet_flux_std < 0.3 * quiet_flux_baseline else 4

            if active_persistence_counter >= dynamic_persistence:
                warning_time = self.steps_data.index[i]
                window_start_idx = i - active_persistence_counter + 1
                window_df = self.steps_data.iloc[window_start_idx:i+1]
                duration_minutes = (window_df.index[-1] - window_df.index[0]).total_seconds() / 60
                peak_flux = window_df['flux'].max()
                flux_integral = window_df['flux'].sum()
                avg_high_energy_flux = window_df['high_energy_flux'].mean()

                cooldown_minutes = min(30, max(10, int(duration_minutes / 2)))
                cooldown_steps = int(cooldown_minutes / 10)

                warning = {
                    'warning_time_utc': warning_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'event_window_start': window_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'event_window_end': warning_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_minutes': round(duration_minutes, 2),
                    'particle_flux_anomaly': f'{anomaly_score:.1f}x background',
                    'peak_flux_in_event_window': round(peak_flux, 2),
                    'flux_integral': round(flux_integral, 2),
                    'high_energy_flux_avg': round(avg_high_energy_flux, 2),
                    'baseline_std': round(quiet_flux_std, 3),
                    'baseline_IQR': round(quiet_flux_iqr, 3),
                    'confidence': min(1.0, (anomaly_score - 1.0) * 0.1 + 0.5)
                }

                self.warnings.append(warning)
                active_persistence_counter = 0
                cooldown_until_index = i + cooldown_steps
