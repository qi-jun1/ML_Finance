
import pandas as pd
import numpy as np 
import numpy as np
import pandas as pd
from typing import List


def cusum_filter_events_dynamic_threshold(
        prices: pd.Series,
        threshold: pd.Series
) -> pd.DatetimeIndex:
    """
    Detect events using the Symmetric Cumulative Sum (CUSUM) filter.

    The Symmetric CUSUM filter is a change-point detection algorithm used to identify events where the price difference
    exceeds a predefined threshold.

    :param prices: A pandas Series of prices.
    :param threshold: A pandas Series containing the predefined threshold values for event detection.
    :return: A pandas DatetimeIndex containing timestamps of detected events.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: 39)
    """
    time_events, shift_positive, shift_negative = [], 0, 0
    price_delta = prices.diff().dropna()
    thresholds = threshold.copy()
    price_delta, thresholds = price_delta.align(thresholds, join="inner", copy=False)

    for (index, value), threshold_ in zip(price_delta.to_dict().items(), thresholds.to_dict().values()):
        shift_positive = max(0, shift_positive + value)
        shift_negative = min(0, shift_negative + value)

        if shift_negative < -threshold_:
            shift_negative = 0
            time_events.append(index)

        elif shift_positive > threshold_:
            shift_positive = 0
            time_events.append(index)

    return pd.DatetimeIndex(time_events)


def daily_volatility_with_log_returns(
        close: pd.Series,
        span: int = 100
) -> pd.Series:
    """
    Calculate the daily volatility at intraday estimation points using Exponentially Weighted Moving Average (EWMA).

    :param close: A pandas Series of daily close prices.
    :param span: The span parameter for the Exponentially Weighted Moving Average (EWMA).
    :return: A pandas Series containing daily volatilities.

    References:
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. (Methodology: Page 44)
    """
    df1 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df1 = df1[df1 > 0]
    df = pd.DataFrame(index=close.index[close.shape[0] - df1.shape[0]:])
 
    df['yesterday'] = close.iloc[df1].values
    df['two_days_ago'] = close.iloc[df1 - 1].values
    returns = np.log(df['yesterday'] / df['two_days_ago'])
    
    stds = returns.ewm(span=span).std().rename("std")

    return stds


def vertical_barrier(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    number_days: int
) -> pd.Series:
    """
    Shows one way to define a vertical barrier.

    :param close: A dataframe of prices and dates.
    :param time_events: A vector of timestamps.
    :param number_days: A number of days for the vertical barrier.
    :return: A pandas series with the timestamps of the vertical barriers.
    """
    timestamp_array = close.index.searchsorted(time_events + pd.Timedelta(days=number_days))
    timestamp_array = timestamp_array[timestamp_array < close.shape[0]]
    timestamp_array = pd.Series(close.index[timestamp_array], index=time_events[:timestamp_array.shape[0]])
    return timestamp_array


def triple_barrier(
    close: pd.Series,
    events: pd.DataFrame,
    profit_taking_stop_loss: list[float, float],
    molecule: list
) -> pd.DataFrame:
    # Filter molecule to ensure all timestamps exist in events
    molecule = [m for m in molecule if m in events.index]

    # Continue with the existing logic
    events_filtered = events.loc[molecule]
    output = events_filtered[['End Time']].copy(deep=True)

    if profit_taking_stop_loss[0] > 0:
        profit_taking = profit_taking_stop_loss[0] * events_filtered['Base Width']
    else:
        profit_taking = pd.Series(index=events.index)

    if profit_taking_stop_loss[1] > 0:
        stop_loss = -profit_taking_stop_loss[1] * events_filtered['Base Width']
    else:
        stop_loss = pd.Series(index=events.index)

    for location, timestamp in events_filtered['End Time'].fillna(close.index[-1]).items():
        df = close[location:timestamp] #takes the price path from start to the vertical barrier:
        df = np.log(df / close[location]) * events_filtered.at[location, 'Side'] #Converts prices to log returns relative to entry, adjusted by Side
        output.loc[location, 'stop_loss'] = df[df < stop_loss[location]].index.min()#earliest time where stop loss is hit within vertical time horizon
        output.loc[location, 'profit_taking'] = df[df > profit_taking[location]].index.min()#earliest time where profit taking is hit within vertical time horizon

    return output


def meta_events(
    close: pd.Series,
    time_events: pd.DatetimeIndex,
    ptsl: List[float],
    target: pd.Series,
    return_min: float,
    num_threads: int,
    timestamp: pd.Series = False,
    side: pd.Series = None
) -> pd.DataFrame:
    # Filter target by time_events and return_min
    target = target.loc[time_events]
    target = target[target > return_min]

    # Ensure timestamp is correctly initialized
    if timestamp is False:
        timestamp = pd.Series(pd.NaT, index=time_events)
    else:
        #set timestamps to events start date.
        timestamp = timestamp.loc[time_events]

    if side is None:
        #if none. side_position is filled entirely with one, so we always go long at every event.
        #both profit and loss barrier is set to the same value.
        side_position, profit_loss = pd.Series(1., index=target.index), [ptsl[0], ptsl[0]]
    else:
        #if side is set then side_position is either 1 for long or -1 for short. 
        #profit and loss barrier is set to the same value.
        side_position, profit_loss = side.loc[target.index], ptsl[:2]

    # Include 'target' and 'timestamp' in the events DataFrame
    events = pd.concat({'End Time': timestamp, 'Base Width': target, 'Side': side_position, 'target': target, 'timestamp': timestamp}, axis=1).dropna(subset=['Base Width'])

    df0 = list(map(
        triple_barrier,
        [close] * num_threads,
        [events] * num_threads,
        [profit_loss] * num_threads,
        np.array_split(time_events, num_threads)
    ))
    df0 = pd.concat(df0, axis=0)

    #set End Time to earliest barrier hit.
    events['End Time'] = df0.dropna(how='all').min(axis=1)

    if side is None:
        events = events.drop('Side', axis=1)

    # Return events including the 'target' and 'timestamp' columns
    return events , df0


def triple_barrier_labeling(
    events: pd.DataFrame,
    close: pd.Series
) -> pd.DataFrame:
    
    events_filtered = events.dropna(subset=['End Time'])
    all_dates = events_filtered.index.union(events_filtered['End Time'].values).drop_duplicates()
    close_filtered = close.reindex(all_dates, method='bfill')
    out = pd.DataFrame(index=events_filtered.index)
    
    #time of first barrier hit
    out['End Time'] = events['End Time']

    #return timestamp
    out["timestamp"] = events["timestamp"]

    #Set the Side to either 1/-1 (long/short) depending on wether the stock price difference is positive (long) or negative (short)
    out['Side'] =  np.sign(close_filtered.loc[events_filtered['End Time'].values].values - close_filtered.loc[events_filtered.index])
    
    return out

def meta_labeling(
    events: pd.DataFrame,
    close: pd.Series
) -> pd.DataFrame:
    """
    Expands label to incorporate meta-labeling.

    :param events: DataFrame with timestamp of vertical barrier and unit width of the horizontal barriers.
    :param close: Series of close prices with date indices.
    :return: DataFrame containing the return and binary labels for each event.

    Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 51
    """
    events_filtered = events.dropna(subset=['End Time'])
    all_dates = events_filtered.index.union(events_filtered['End Time'].values).drop_duplicates()
    close_filtered = close.reindex(all_dates, method='bfill')
    out = pd.DataFrame(index=events_filtered.index)
    
    #time of first barrier hit
    #out['End Time'] = events['End Time']

    #Return of Label: close price at end time/ close price at start time - 1.
    out['Return of Label'] = close_filtered.loc[events_filtered['End Time'].values].values / close_filtered.loc[events_filtered.index] - 1

    #timestamp contains original vertical barrier dates per event. The equality returns zero if unequal so pt or sl
    # is hit or 1 if vertical barrier is hit. So Label = 1 if a barrier is hit and 0 if vertical barrier is hit.    
    if 'Side' in events_filtered:
        out['Return of Label'] *= events_filtered['Side']
    out['Label'] = np.sign(out['Return of Label'])  * (1 - (events['End Time'] == events['timestamp']))
    if 'Side' in events_filtered:
        out.loc[out['Return of Label'] <= 0, 'Label'] = 0
        #out['Side'] = events_filtered['Side']
    return out

