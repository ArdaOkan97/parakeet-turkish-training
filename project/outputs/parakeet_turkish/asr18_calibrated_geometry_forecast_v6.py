"""Explicit<=40 same-process prospective extension of the measured20 policy.

The geometry, warmup, running-high-water margins and first-native startup
limits are unchanged. Allowing the SECOND resident window is a new finite
numerical/resource trial, not proof of40-update stability or production fit.
"""
from asr18_calibrated_geometry_forecast_v5 import CalibratedGeometryForecast as Previous
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest


class CalibratedGeometryForecast(Previous):
    def budget(self, *, rows, current_private_bytes, update_in_process):
        require(type(update_in_process) is int and 1 <= update_in_process <= 40,
                'At most two20-update resident windows; no unbounded native loop')
        # The inherited formula has exactly two branches: index1 warmup, every
        # later index running high-water. Its explicit20 guard is preserved in
        # that source; index2 selects the identical later-index formula here.
        value = super().budget(rows=rows, current_private_bytes=current_private_bytes,
                               update_in_process=1 if update_in_process == 1 else 2)
        value.update(kind='ASR18_finite_measured_geometry_native_forecast_v6', source=record(__file__),
                     actual_update_index_in_this_process=update_in_process, maximum_native_updates_in_process=40,
                     second_resident_window_is_new_unproven_scope=True)
        return value

    def state_dict(self):
        value = super().state_dict()
        value.update(kind='ASR18_finite_native_forecast_high_water_v6', source=record(__file__))
        return value

    def load_state_dict(self, state):
        expected = self.state_dict()
        require(type(state) is dict and set(state) == set(expected)
                and digest({key: value for key, value in state.items() if key != 'running_native_high_water_bytes'})
                == digest({key: value for key, value in expected.items() if key != 'running_native_high_water_bytes'})
                and type(state['running_native_high_water_bytes']) is int
                and state['running_native_high_water_bytes'] >= self.calibrated_peak,
                'Exact source/calibration and monotonic actual native high-water required')
        self.high_water = max(self.high_water, state['running_native_high_water_bytes'])
