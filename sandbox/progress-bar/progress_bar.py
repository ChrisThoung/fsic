# -*- coding: utf-8 -*-
"""
progress_bar
============
Experiments extending [`tqdm`](https://github.com/tqdm/tqdm)-based progress
bars, to eventually augment the package version of `ProgressBarMixin`(?).

This module defines the augmented `ProgressBarMixin`.

The original version of `ProgressBarMixin` (implemented in fsic version 0.8.0)
over-rode `iter_periods()` by adding a single keyword argument,
`progress_bar`. If `True`, the over-riding method wraps the `PeriodIter` object
with `tqdm`. Looping through the periods for solution then advances the
progress bar.

Currently, `progress_bar=True` just creates a standard `tqdm` progress bar. Any
further customisation needs custom code, rather than what's currently available
in `progress_bar=True`. For example, customisation would need the user to write
something like:

    model = ...

    period_iter = model.iter_periods()

    with tqdm(period_iter) as pbar:
        for t, period in period_iter:
            pbar.set_description(str(period))
            model.solve_t(t)
            pbar.update(1)

        pbar.set_description('DONE')

The generalised implementation below modifies the mixin to accept a wider range
of arguments to run the progress bar, without having to resort to code like the
above. The new prototype `iter_periods()` method preserves the original method
definition but the `progress_bar` parameter now handles:

* bool:
    * `False`: Do nothing
    * `True`:  Use the default progress bar, `TqdmFsicStandard` (this can be
               over-ridden)
* str: The name of a valid progress bar class (which must match the name of an
       attribute of the current object: various options are available in the
       class already; more can be added)
* class: A progress bar class that wraps an iterable in the same way as `tqdm`

The above requires a range of ways of handling the argument, which needs
reviewing before deciding if this is the best way to extend
`ProgressBarMixin`.

One consequence, for example, is the inability to *remove* pre-existing
progress bar classes. Instead, the only solution would be to replace the
attribute with either `None` - which is currently unhandled - or some other
placeholder class. Ideally, anything along these lines would be a runtime
problem when enabling a progress bar, rather than at import.

Unresolved issues:

1. Is the implementation of progress bar classes as class-level attributes
   problematic in any way?
2. Should `iter_periods()` really handle all these different cases?
3. Are there any performance issues with the implementation, relative to just
   wrapping with `tqdm`?
"""

import copy
from typing import Any, Hashable, Iterable, Optional, Tuple, Union

from fsic.core import PeriodIter


try:
    from tqdm import tqdm as _tqdm

except ModuleNotFoundError:
    # If `tqdm` is not installed, replace with a class that just raises a
    # `ModuleNotFoundError`
    # By this approach, the code runs normally if, in `iter_periods()` below,
    # `progress_bar=False`
    class _tqdm:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                '`ProgressBarMixin` requires `tqdm` to be installed'
            )


TqdmFsicBasic = _tqdm  # Alias for `tqdm`


class _TqdmFsicBase(_tqdm):
    """Base class for fsic `tqdm`-based progress bars.

    This implementation follows the example set out in:
    https://github.com/tqdm/tqdm?tab=readme-ov-file#description-and-additional-stats
    """

    def __init__(
        self,
        period_iter: Union[PeriodIter, Iterable[Tuple[int, Hashable]]],
        *args: Any,
        bar_format: Optional[str] = None,
        **kwargs: Any,
    ):
        # The line below converts a `PeriodIter`-like of (index, period) pairs
        # to two separate lists: one of indexes only and one of periods only
        # For example, from:
        #     (1, 1990), (2, 1991), (3, 1992), ...
        # to:
        #     [1, 2, 3, ...], [1990, 1991, 1992, ...]
        self.indexes, self.periods = map(list, zip(*list(copy.deepcopy(period_iter))))
        super().__init__(period_iter, *args, bar_format=bar_format, **kwargs)

    @property
    def format_dict(self):
        """`tqdm` dict extended with the most recent index and period solved."""
        d = super().format_dict
        n = d['n']  # `tqdm` index of the iteration just completed

        if n < len(self.indexes):
            # Get most recent index/period solved
            index = self.indexes[n]
            period = self.periods[n]
        else:
            # Use 'DONE' for overflow (end)
            index = period = 'DONE'

        d.update(index=index, period=period)
        return d


class TqdmFsicStandard(_TqdmFsicBase):
    """Standard fsic progress bar reporting the period and percentage just solved."""

    def __init__(
        self,
        *args: Any,
        bar_format: Optional[str] = '{period}:{percentage:3.0f}%|{bar}{r_bar}',
        **kwargs: Any,
    ):
        super().__init__(*args, bar_format=bar_format, **kwargs)


class ProgressBarMixin:
    """Mixin to add an (optional) `tqdm`-based progress bar during solution.

    **Requires `tqdm`** if the progress bar is turned on. With the progress bar
    turned off (which is the default), the code will run without needing
    `tqdm`.

    Examples
    --------
    Add the mixin to the model class definition:

        from fsic import BaseModel
        from fsic.extensions import ProgressBarMixin

        class ExampleModel(ProgressBarMixin, BaseModel):
            ...

    By default, behaviour is unchanged:

    >>> model = ExampleModel(range(10))
    >>> model.solve()

    Passing `progress_bar=True` will print a customised `tqdm`-based progress
    bar (`TqdmFsicStandard`) to the screen:
    >>> model.solve(progress_bar=True)

    Other progress bars (if defined as class attributes) are accessible by
    name, as strings:
    >>> model.solve(progress_bar='TqdmFsicBasic')

    Or you can pass progress bar classes as arguments:

    # Pre-defined class, built in to `ProgressBarMixin`
    >>> model.solve(progress_bar=ProgressBarMixin.TqdmFsicStandard)

    # Using a `tqdm`-compatible class (just `tqdm` for ease of demonstration)
    >>> from tqdm import tqdm
    >>> model.solve(progress_bar=tqdm)

    Notes
    -----
    This mixin:
     - defines:
        - `PROGRESS_BAR_DEFAULT`, a class-level attribute that sets the
          preferred progress bar implementation if `progress_bar=True` in
          `iter_periods()` (as below)
     - wraps:
        - `iter_periods()`, to handle a new `progress_bar` parameter
     - adds the following classes as attributes (to implement built-in progress
       bar classes; an approach which needs careful review before integrating):
        - TqdmFsicBasic: An alias for the standard `tqdm` progress bar
        - TqdmFsicStandard: The default progress bar, which prints the last
                            period solved, along with the percentage

    In implementation, the mixin intervenes by over-riding `iter_periods()` to
    (optionally) wrap the `PeriodIter` in a `tqdm`-based class. To be flexible
    to different progress bar options, the `progress_bar` argument to
    `iter_periods()` can be any one of:

    * bool:
        * `False`: Do nothing
        * `True`:  Use the default progress bar, `TqdmFsicStandard` (this can be
                   over-ridden)
    * str: The name of a valid progress bar class (which must match the name of
           an attribute of the current object: various options are available in
           the class already; more can be added)
    * class: A progress bar class that wraps an iterable in the same way as
             `tqdm`

    To use alternative progress bars, either:

    1. Pass a progress bar class as the `progress_bar` argument e.g. a
       user-defined one, `tqdm` or one of the pre-defined classes in this
       module such as `TqdmFsicBasic`/`ProgressBarMixin.TqdmFsicBasic` or
       `TqdmFsicStandard`/`ProgressBarMixin.TqdmFsicStandard`
    2. Add a progress bar class as a class-level attribute (as for
       `TqdmFsicBasic` and `TqdmFsicStandard` in this class) and then use
       `progress_bar` to pass the class's name as a string

    To control the default progress bar, if `progress_bar` is `True`, set the
    value of `PROGRESS_BAR_DEFAULT` to either:

    * The name of the preferred progress bar class (i.e. a str), which must be
      bound as a class attribute (as [2] above)
    * A progress bar class to use as the default
    """

    # Define progress bars as class-level attributes. `iter_periods()` can then
    # access these by name (str) as needed
    TqdmFsicBasic: 'tqdm' = TqdmFsicBasic        # noqa: F821
    TqdmFsicStandard: 'tqdm' = TqdmFsicStandard  # noqa: F821

    # Set the default progress bar (e.g. if `progress_bar=True`) as either a
    # class name (str) or class
    PROGRESS_BAR_DEFAULT: Union[str, 'tqdm'] = TqdmFsicStandard  # noqa: F821

    def iter_periods(
        self,
        *args,
        progress_bar: Union[bool, str, 'tqdm'] = False,  # noqa: F821
        **kwargs,
    ) -> Iterable[Tuple[int, Hashable]]:
        """Modified `iter_periods()` method: Display a progress bar if `progress_bar` is not `False` (see class definition notes and below).

        **Requires `tqdm`** if the progress bar is turned enabled. With the
        progress bar disabled (which is the default), the code will run fine,
        without needing `tqdm`.

        Parameters
        ----------
        progress_bar :
            Depending on the argument type:
             - `bool`: If `False`, return the `PeriodIter` object as usual
                       i.e. with no wrapper; if `True` use the default progress
                       bar (as set by `PROGRESS_BAR_DEFAULT`)
             - `str`:  Name of an existing progress bar class (i.e. bound to
                       the current object)
             - class:  A progress bar class (matching the `tqdm` API)

        *args, **kwargs :
            Arguments to pass to the base class `iter_periods()` method
        """
        # Get the original `PeriodIter` object
        period_iter = super().iter_periods(*args, **kwargs)

        # Optionally wrap the iterator with a progress bar
        if progress_bar:
            # Get the default setting if `progress_bar=True`
            if isinstance(progress_bar, bool):
                progress_bar = self.PROGRESS_BAR_DEFAULT

            # Handle the `progress_bar` argument as either...
            if isinstance(progress_bar, str):
                # ...the progress bar name: get the corresponding class
                wrapper = getattr(self, progress_bar)
            else:
                # ...a class directly
                wrapper = progress_bar

            period_iter = wrapper(period_iter)

        return period_iter
