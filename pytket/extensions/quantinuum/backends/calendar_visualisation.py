# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# original source: https://github.com/meta4/mplcal.git
# Copyright (c) 2020, Peter Wilson
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Submodule providing calendar visualisation functionality"""

import calendar
import datetime

import matplotlib.pyplot as mpl
import numpy as np
from matplotlib.figure import Figure

calendar.setfirstweekday(0)


class QuantinuumCalendar:
    """Calendar visualisation using matplotlib. The calendar
    is rendered for a specified month and year.

    """

    @property
    def months(self) -> list[str]:
        return [calendar.month_name[i] for i in range(1, 13)]

    @property
    def weekdays(self) -> list[str]:
        return [calendar.day_name[i] for i in range(7)]

    def __init__(self, year: int, month: int, title_prefix: str):
        """Construct a `QuantinuumCalendar` object.

        :param year: An integer representing the year, i.e. 2024.
        :param month: An integer representing the month in the year,
            i.e. 2 for February.
        :param title_prefix: A prefix to add to the title
        """
        self._title: str = f"{title_prefix} Operations Calendar\n\
            {self.months[month - 1]} {year}\
            ({datetime.datetime.now().astimezone().strftime('%Z')})"
        self._cal: np.ndarray = np.asarray(calendar.monthcalendar(year, month))
        self._events: np.ndarray = np.full(
            self._cal.shape, fill_value=None, dtype=object
        )
        self._colors: np.ndarray = np.full(
            self._cal.shape, fill_value=None, dtype=object
        )

    def _add_event(self, day: int, event_str: str) -> None:
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
                self._events[week, week_day] = f"{event_str1}\n\n{event_str}"
            self._colors[week, week_day] = "mistyrose"
        except RuntimeError:
            raise RuntimeError("Day outside of specified month")  # noqa: B904

    def add_events(self, events_list: list[dict[str, object]]) -> None:
        """Add list of events. Each event is a dictionary and
        must have the following keys:
        * 'start-date', a datetime.datetime object
        * 'end-date', a datetime.datetime object
        * 'event-type', a string specifying if the device is `online`
            or has a `reservation`.
        * 'organization', a string specifying the organisation with
            reservation access. Otherwise, if the event-type is `online`,
            the organization is listed as `fairshare`.
        """
        for event in events_list:
            event_start: datetime.datetime = event["start-date"]  # type: ignore
            event_end: datetime.datetime = event["end-date"]  # type: ignore
            event_type: str = event["event-type"]  # type: ignore
            dt_format = "%H:%M"
            duration = (event_end - event_start).seconds / 3600
            event_str = (
                f"{event_type}\nStart: {event_start.strftime(dt_format)} ({duration}h)"
            )
            day = event_start.day
            self._add_event(day, event_str)

    def build_calendar(
        self,
        figsize: tuple[float, float] = (40, 20),
        fontsize: float = 15,
        titlesize: float = 40,
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
