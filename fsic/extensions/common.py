# -*- coding: utf-8 -*-
"""
Mixins applicable to both the model and linker classes.
"""

class ProgressBarMixin:

    def iter_periods(self, *args, show_progress: bool = False, **kwargs):
        """Modified `iter_periods()` method: Display a `tqdm` progress bar if `show_progress=True`. **Requires `tqdm`**"""
        # Get the original `PeriodIter` object
        period_iter = super().iter_periods(*args, **kwargs)

        # Optionally wrap with `tqdm`
        if show_progress:
            from tqdm import tqdm
            period_iter = tqdm(period_iter)

        return period_iter
