"""Submodule providing calendar visualisation functionality"""

from typing import List, Dict, NoReturn, Tuple
import calendar
import datetime

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.figure import Figure

calendar.setfirstweekday(0)


class QuantinuumCalendar(object):
    """Calendar visualisation using matplotlib. The calendar
       is rendered for a specified month and year.
    
    """

    @property
    def months(self) -> List[str]:
        return [calendar.month_name[i] for i in range(1, 13)]

    @property
    def weekdays(self) -> List[str]:
        return [calendar.day_name[i] for i in range(7)]

    def __init__(self, year: int, month: int, title_prefix: str):
        """Construct a `QuantinuumCalendar` object.

        :param year: An integer representing the year, i.e. 2024.
        :param month: An integer representing the month in the year,
            i.e. 2 for February.
        :param title_prefix: A prefix to add to the title
        """
        self._title: str = f"{title_prefix} Operations Calendar\n{self.months[month-1]} {year} ({datetime.datetime.now().astimezone().strftime('%Z')})"
        self._cal: np.ndarray = np.asarray(calendar.monthcalendar(year, month))
        self._events: np.ndarray = np.full(
            self._cal.shape, fill_value=None, dtype=object
        )
        self._colors: np.ndarray = np.full(
            self._cal.shape, fill_value=None, dtype=object
        )

    def _add_event(self, day: int, event_str: str):
        """Add event.

        :param day: An integer specifing day in the month
        :param event_str: A string containing the event description.
        """

        indices = np.where(self._cal == day)
        week = indices[0][0]
        week_day = indices[1][0]
        try:
            if self._events[week, week_day] is None:
                self._events[week, week_day] = event_str
            else:
                event_str1 = self._events[week, week_day]
                self._events[week, week_day] = f"{event_str1}\n{event_str}"
            self._colors[week, week_day] = "mistyrose"
        except RuntimeError:
            raise RuntimeError(f"Day outside of specified month")

    def add_events(self, events_list: List[Dict[str, object]]) -> NoReturn:
        """Add list of events. Each event is a dictionary and
        must have the following keys:
        * 'start-date', a datetime.datetime object
        * 'end-date', a datetime.datetime object
        * 'event-type', a string specifying if the device is `online` or reserved.
        * 'organization', a string specifying the organisation with reservation
            access. Otherwise, if the event-type is `online`, the organization
            is listed as `fairshare`.
        """
        for event in events_list:
            event_start: datetime.datetime = event["start-date"]
            event_end: datetime.datetime = event["end-date"]
            event_type: str = event["event-type"]
            event_org: str = event["organization"]
            dt_format = f"%H:%M"
            duration = (event_end - event_start).seconds / 3600
            event_str = f"{event_type}-{event_org}\nStart: {event_start.strftime(dt_format)} ({duration}h)"
            day = event_start.day
            self._add_event(day, event_str)

    def build_calendar(
        self,
        figsize: Tuple[float, float] = (40, 20),
        fontsize=15,
        titlesize=40,
    ) -> Figure:
        """Display calendar on a matplotlib.figure.Figure object.

        :param figsize:
        :param fontsize:
        :param titlesize:
        :returns: A matplotlib visualisation of a calendar with events
        :return_type matplotlib.figure.Figure:
        """
        "Create the calendar figure"
        f, axes = mpl.subplots(
            len(self._cal),
            7,
            sharex=True,
            sharey=True,
            figsize=figsize,
            dpi=80,
        )
        for week, ax_row in enumerate(axes):
            for week_day, ax in enumerate(ax_row):
                ax.set_xticks([])
                ax.set_yticks([])
                if self._colors[week, week_day] is not None:
                    ax.set_facecolor(self._colors[week][week_day])
                if self._cal[week][week_day] != 0:
                    ax.text(
                        0.02,
                        0.98,
                        str(self._cal[week][week_day]),
                        verticalalignment="top",
                        horizontalalignment="left",
                        fontsize=fontsize,
                    )
                ax.text(
                    0.03,
                    0.85,
                    self._events[week][week_day],
                    verticalalignment="top",
                    horizontalalignment="left",
                    fontsize=fontsize,
                )

        # use the titles of the first row as the weekdays
        for n, day in enumerate(self.weekdays):
            axes[0][n].set_title(day, fontsize=fontsize)

        # Place subplots in a close grid
        f.subplots_adjust(hspace=0, wspace=0)
        f.suptitle(self._title, fontsize=titlesize, fontweight="bold")
        return f
