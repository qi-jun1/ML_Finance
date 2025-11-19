#/usr/

import argparse
from collections import namedtuple
import datetime
import numpy as np
import operator #and_
import os
import pandas as pd
import pickle
import seaborn
import sys
import util

isAlmostZero = lambda x: abs(x) < util.almostZero
nan_2_almostZero = lambda x: util.almostZero if np.isnan(x) else x
nan_2_neg_inf = lambda x: -np.inf if np.isnan(x) else x
post_20cm_start = lambda x: x > datetime.date(2020, 8, 24)
__ZTBThres = {'10cm': 0.0988, '20cm': 0.1988}



def _IsNotST(inputDf):
    st = (inputDf.st > 0)
    amount = inputDf.amount.map(lambda x: not np.isnan(x))
    return (st * amount).map(int)

def GenerateNonSTLabel(dictByDate, dictBySymbol):
    for date, v in dictByDate.items():
        #print('Processing st label for %s' % date)
        isNotST = _IsNotST(v)
        dictByDate[date]['isNotST'] = isNotST
    for symbol, v in dictBySymbol.items():
        #print('Processing st label for %s' % symbol)
        isNotST = _IsNotST(v)
        dictBySymbol[symbol]['isNotST'] = isNotST

def GenerateReturns(symbols,dictBySymbol, dictByDate):
    symbols = list(dictBySymbol.keys())
    dates   = dictBySymbol[symbols[0]].index

    # 1) 构造面板
    close_df    = pd.DataFrame({s: df['close']    for s, df in dictBySymbol.items()}, index=dates)
    open_df     = pd.DataFrame({s: df['open']     for s, df in dictBySymbol.items()}, index=dates)
    turnover_df = pd.DataFrame({s: df['turnover'] for s, df in dictBySymbol.items()}, index=dates)

    # 2) 计算各种 returns
    returns_df    = close_df.pct_change(fill_method=None)
    returns_co_df = open_df.div(close_df.shift(1)) - 1
    returns_oc_df = close_df.div(open_df) - 1

    # 3) returns_20cm 标记
    cutoff = datetime.date(2020, 8, 24)
    is20cm_dates = (pd.Series(dates) > cutoff).astype(int).values  # length = len(dates)
    mask300_688  = np.array([1 if (s.startswith('300') or s.startswith('688')) else 0 
                              for s in symbols])                       # length = len(symbols)
    returns_20cm = pd.DataFrame(
        np.outer(is20cm_dates, mask300_688),
        index=dates, columns=symbols
    )

    # 4) 计算 isZTB / isDTB
    th10, th20 = __ZTBThres['10cm'], __ZTBThres['20cm']
    cond20 = returns_20cm.astype(bool)
    isZTB = pd.DataFrame(
        np.where(cond20, returns_df >= th20, returns_df >= th10),
        index=dates, columns=symbols
    ).astype(int)
    isDTB = pd.DataFrame(
        np.where(cond20, returns_df <= -th20, returns_df <= -th10),
        index=dates, columns=symbols
    ).astype(int)

    # 5) turnover_adj：如果涨停 && turnover<3%，就设为3%
    turnover_adj = turnover_df.mask((isZTB == 1) & (turnover_df < 3.0), 3.0)

    # 6) 回写到 dictBySymbol
    for s in symbols:
        df = dictBySymbol[s]
        df['returns']       = returns_df   [s].values
        df['returns_co']    = returns_co_df[s].values
        df['returns_oc']    = returns_oc_df[s].values
        df['returns_20cm']  = returns_20cm [s].values
        df['turnover_adj']  = turnover_adj [s].values
        df['isZTB']         = isZTB        [s].values
        df['isDTB']         = isDTB        [s].values
        dictBySymbol[s] = df

    # 7) 回写到 dictByDate （只写 returns，其他按需补充同理）
    for date in dates:
        day_df = dictByDate[date]
        day_df['returns'] = returns_df.loc[date].values
        dictByDate[date] = day_df

    return dictBySymbol, dictByDate

def GenerateFeaturesBySymbol(symbols,dictBySymbol):
    """
    向量化生成特征，替代原来的逐 symbol 循环，包括：
      - MA(5,10,20,34,55,233) of close
      - BIAS(6) for close/high/low
      - BIAS2(5) for high/low
      - Bollinger Bands (20) and %B
      - RSV (10,20)
      - Aroon Up/Down (10,20)
      - MACD (DIF, DEA, MACD) with spans (10,20,7) + cross
      - Slopes: MA5/10/20 diff
    返回更新后的 dictBySymbol（原地修改并返回）。
    """
    symbols = list(dictBySymbol.keys())
    dates   = dictBySymbol[symbols[0]].index

    # 1) 面板化 OHLC
    close = pd.DataFrame({s: df['close'] for s, df in dictBySymbol.items()}, index=dates)
    high  = pd.DataFrame({s: df['high']  for s, df in dictBySymbol.items()}, index=dates)
    low   = pd.DataFrame({s: df['low']   for s, df in dictBySymbol.items()}, index=dates)

    # 2) MA Features
    ma_windows = [5, 10, 20, 34, 55, 233]
    ma_features = {
        f'MA{w}_close': close.rolling(w).mean()
        for w in ma_windows
    }

    # 3) BIAS Features (window=6)
    w_bias = 6
    ma6_close = close.rolling(w_bias).mean()
    bias_features = {
        'BIAS_6_close': (close - ma6_close) / ma6_close,
        'BIAS_6_high':  (high  - high.rolling(w_bias).mean()) / high.rolling(w_bias).mean(),
        'BIAS_6_low':   (low   - low.rolling(w_bias).mean()) / low.rolling(w_bias).mean(),
    }

    # 4) BIAS2 Features (window=5 pro-forma)
    w2 = 5
    bias2_features = {
        'BIAS2_5_high': (high - high.rolling(w2).mean()) / high.rolling(w2).mean(),
        'BIAS2_5_low':  (low  - low.rolling(w2).mean()) / low.rolling(w2).mean(),
    }

    # 5) Bollinger Bands (20)
    m20 = close.rolling(20).mean()
    s20 = close.rolling(20).std()
    bb_features = {
        'BB_mid_20':   m20,
        'BB_upper_20': m20 + 2*s20,
        'BB_lower_20': m20 - 2*s20,
        'BB_percent_20': (close - (m20 - 2*s20)) / (4*s20).replace(0, np.nan),
    }

    # 6) RSV (10,20)
    rsv_features = {}
    for w in (10, 20):
        hh = high.rolling(w).max()
        ll = low.rolling(w).min()
        rsv_features[f'RSV_{w}'] = (close - ll) / (hh - ll).replace(0, np.nan)

    # 7) Aroon Up/Down (10,20)
    def aroon_up(arr, w):
        idx = np.argmax(arr)
        return (w - 1 - idx) / (w - 1) * 100
    def aroon_dn(arr, w):
        idx = np.argmin(arr)
        return (w - 1 - idx) / (w - 1) * 100

    aroon_features = {}
    for w in (10, 20):
        aroon_features[f'AroonUp_{w}'] = high.rolling(w).apply(
            lambda arr: aroon_up(arr, w), raw=True
        )
        aroon_features[f'AroonDn_{w}'] = low.rolling(w).apply(
            lambda arr: aroon_dn(arr, w), raw=True
        )

    # 8) MACD (10,20,7) + Cross
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    DIF   = ema10 - ema20
    DEA   = DIF.ewm(span=7, adjust=False).mean()
    MACD  = DIF - DEA
    prev_diff = (DIF - DEA).shift(1)
    diff      = DIF - DEA
    cross     = pd.DataFrame(np.nan, index=diff.index, columns=diff.columns)
    cross[(diff > 0) & (prev_diff <= 0)] = 1
    cross[(diff < 0) & (prev_diff >= 0)] = -1
    
    macd_features = {
        'DIF': DIF,
        'DEA': DEA,
        'MACD': MACD,
        'Cross_MACD': cross,
    }

    # 9) Slopes of MA5/10/20
    slope_features = {}
    for w in [5, 10, 20, 34, 55, 233]:
        ma_col = f'MA{w}_close'        # 之前计算的均线列名
        slope_col = f'slope_ma{w}'     # 原排名函数期待的列名
        # 计算斜率：直接用差分
        slope_features[slope_col] = ma_features[ma_col].diff()
    slope_features['slope_DEA'] = DEA.diff()
    slope_features['slope_DIF'] = DIF.diff()
    # 10) 合并所有特征：
    all_feats = {}
    for d in (ma_features, bias_features, bias2_features,
              bb_features, rsv_features, aroon_features,
              macd_features, slope_features):   # 用新的 slope_features
        all_feats.update(d)

    # 11) 回写到 dictBySymbol
    for sym in symbols:
        df = dictBySymbol[sym]
        for name, feat_df in all_feats.items():
            df[name] = feat_df[sym].values
        dictBySymbol[sym] = df

    return dictBySymbol

def transferZScore(x: pd.Series) -> pd.Series:
    """Vectorized Fermi function normalization."""
    xt = x.clip(-99, 99)
    exp_xt = np.exp(xt)
    return (exp_xt - 1) / (exp_xt + 1)

def _RankSymbolsByCloseTrend(dictBySymbol, asOfDate, numDays=10):
    print(f"=== Ranking by CloseTrend on {asOfDate} ===")


    dates = util.GetLastEquityTradingDates(asOfDate, numDays)

    symbols = list(dictBySymbol.keys())

    close_sub  = close_panel.loc[dates] 
    ma_subs    = {w: df.loc[dates] for w, df in ma_dfs.items()}
    slope_subs = {w: df.loc[dates] for w, df in slope_dfs.items()}
    isNotST    = pd.Series({
        s: int(dictBySymbol[s].loc[dates, 'isNotST'].all())
        for s in symbols
    })
    pairs = [
        (5,10),(5,20),(5,34),(5,55),(5,233),
        (10,20),(10,34),(10,55),(10,233),
        (20,34),(20,55),(20,233),
        (34,55),(34,233),(55,233)
    ]

    ma_vs_ma = np.stack([
        (ma_subs[i] > ma_subs[j]).values
        for i, j in pairs
    ], axis=0)
    ma_vs_ma_scores = ma_vs_ma.mean(axis=1)

    close_vs_ma = np.stack([
        (close_sub > ma_subs[w]).values
        for w in ma_windows
    ], axis=0)  # (6, numDays, numSymbols)
    close_vs_ma_scores = close_vs_ma.mean(axis=1)  # (6, numSymbols)
        
    slope_pos = np.stack([
        (slope_subs[w] > 0).values
        for w in ma_windows
    ], axis=0)  # (6, numDays, numSymbols)
    slope_pos_scores = slope_pos.mean(axis=1)  # (6, numSymbols)

        
    trend1 = ma_vs_ma_scores.mean(axis=0)    # (numSymbols,)
    trend2 = close_vs_ma_scores.mean(axis=0)
    trend3 = slope_pos_scores.mean(axis=0)
    combined    = np.vstack([trend1, trend2, trend3]).T  # (numSymbols, 3)
    final_scores = combined.mean(axis=1) * isNotST.values
    final_series = pd.Series(final_scores, index=symbols)


    final_ranks = final_series.rank(ascending=False, method='first')

    return {'trnd_score': final_ranks}, {'trnd_score': final_series}

def _RankSymbolsByTurnover(dictBySymbol, asOfDate, numDays=10): 
    print(f"=== Ranking by Turnover on {asOfDate} ===")
    precision = util.precisionError
    idx     = date2idx[asOfDate]
    symbols = list(dictBySymbol.keys())

    win_tos = turnover_panel.values[idx-numDays+1:idx+1, :]  # (N, M)
    win_ret = returns_panel.values [idx-numDays+1:idx+1, :]  # (N, M)
    win_tos = np.nan_to_num(win_tos, nan=precision)
    win_ret = np.nan_to_num(win_ret, nan=precision)

    x   = win_tos
    out = np.zeros_like(x)
    mask = (x >= 0) & (x <= 2)
    out[mask] = x[mask] * 20
    mask = (x > 2) & (x <= 3)
    out[mask] = np.ceil((x[mask] - 2) * 10) + 40
    mask = (x > 3) & (x <= 5)
    out[mask] = np.ceil((x[mask] - 3) * 5) + 50
    mask = (x > 5) & (x <= 15)
    out[mask] = np.ceil((x[mask] - 5) * 4) + 60
    mask = (x > 15) & (x <= 30)
    out[mask] = 100.0
    mask = (x > 30) & (x <= 50)
    out[mask] = 100.0 - np.floor((x[mask] - 30) * 5)
    toScores = out  # (N, M)

    isNotST_mask = isNotST_array[idx]  # 预先计算好的非 ST 掩码

    thr = [50, 80, 90, 99]
    level_scores = np.mean([ (toScores > t).sum(axis=0) / numDays for t in thr ], axis=0)
    level_scores *= isNotST_mask

    # 3) Z‐score
    mu    = toScores.mean(axis=0)
    sigma = toScores.std(axis=0)
    sigma[sigma == 0] = precision
    z     = transferZScore(mu / sigma)

    # 4) 分布分数
    mins = toScores.min(axis=0)
    maxs = toScores.max(axis=0)
    spans = maxs - mins
    spans_safe = spans.copy()
    spans_safe[spans_safe == 0] = precision

    norm1 = ((toScores - mins) / spans_safe) ** 2
    sum1  = norm1.sum(axis=0)
    sum1_safe = sum1.copy()
    sum1_safe[sum1_safe == 0] = precision
    h1 = (1 - ((norm1 / sum1_safe)**2) .sum(axis=0)) / (1 - 1/numDays)

    maxs_safe = maxs.copy()
    maxs_safe[maxs_safe == 0] = precision
    norm2 = (toScores / maxs_safe) ** 2
    sum2  = norm2.sum(axis=0)
    sum2_safe = sum2.copy()
    sum2_safe[sum2_safe == 0] = precision
    h2 = (1 - ((norm2 / sum2_safe)**2) .sum(axis=0)) / (1 - 1/numDays)

    mids      = (maxs + mins) / 2
    mids_safe = mids.copy()
    mids_safe[mids_safe == 0] = precision
    m = 1 - spans_safe / mids_safe
    m[m < 0] = 0

    dist_scores = np.mean([h1, h2, m, z], axis=0) * isNotST_mask

    # 5) 正负调优
    pos = np.where(win_ret >  0, win_tos, 0.0)
    neg = np.where(win_ret <= 0, win_tos, 0.0)

    pos_mean = pos.mean(axis=0)
    neg_mean = np.abs(neg).mean(axis=0)
    denom_m = pos_mean + neg_mean
    denom_m[denom_m == 0] = precision
    mean_score = (pos_mean - neg_mean) / denom_m

    pos_sum = pos.sum(axis=0)
    neg_sum = np.abs(neg).sum(axis=0)
    denom_s = pos_sum + neg_sum
    denom_s[denom_s == 0] = precision
    sum_score = (pos_sum - neg_sum) / denom_s
    sum_score = np.clip(sum_score, None, 1.0)

    posneg_scores = np.mean([mean_score, sum_score], axis=0) * isNotST_mask

    # 6) 打包为 Series 并返回
    level_ser  = pd.Series(level_scores,  index=symbols)
    dist_ser   = pd.Series(dist_scores,   index=symbols)
    posneg_ser = pd.Series(posneg_scores, index=symbols)

    featScores = {
        'to_{dirc}_c':       level_ser,
        'to_{dist}_c':       dist_ser,
        'to_{pos-vs-neg}_c': posneg_ser,
    }
    featRanks = {
        k: v.rank(ascending=False, method='first')
        for k, v in featScores.items()
    }

    return featRanks, featScores

def _RankSymbolsByActiveness(dictBySymbol, asOfDate, numDays=10):
    print(f"=== Ranking by Activeness on {asOfDate} ===")

    idx = date2idx[asOfDate]
    start, end = idx - numDays + 1, idx + 1
    
    win = isZTB_panel.iloc[start:end, :]  # shape (numDays, M)

    freq = win.sum(axis=0) / numDays  # (M,)

    scores = freq * isNotST_mask            # (M,)
    ranks  = pd.Series(scores, index=symbols) \
                     .rank(ascending=False, method='first')

    featRanks  = {'actv_{%ZTB}_c': ranks}
    featScores = {'actv_{%ZTB}_c': pd.Series(scores, index=symbols)}
    return featRanks, featScores

def _RankSymbolsByReturns(dictBySymbol, asOfDate, numDays=10):
    print(f"=== Ranking by Returns on {asOfDate} ===")

    dates_window = util.GetLastEquityTradingDates(asOfDate, numDays)

    idx = date2idx[asOfDate]
    start = idx - numDays + 1

    R_df  = returns_panel.loc[dates_window]      # DataFrame
    RO_df = returns_oc_panel.loc[dates_window]
    Z_df  = isZTB_panel.loc[dates_window]
    D_df  = isDTB_panel.loc[dates_window]

    R  = R_df.values      # shape (numDays, S)
    RO = RO_df.values
    Z  = Z_df.values
    D  = D_df.values

    thr = np.array([0.03, 0.05, 0.08])[:, None]        # (3,1)
    absR = np.abs(R)                                   # (N,M)
    cond = (absR[None, :, :] <= thr[:, None, :])       # (3,N,M)
    lvl = cond.sum(axis=1) / numDays                   # (3,M)
    level_scores = lvl.mean(axis=0) * isNotST_mask     # (M,)

    mu    = absR.mean(axis=0)
    sigma = absR.std(axis=0).clip(min=precision)
    x     = (mu / sigma).clip(-99, 99)
    zscr  = (np.exp(x) - 1)/(np.exp(x) + 1)

    mins  = absR.min(axis=0)
    maxs  = absR.max(axis=0)
    spans = (maxs - mins).clip(min=precision)

    norm1 = ((absR - mins) / spans)**2
    sum1 = norm1.sum(axis=0)
    sum1_safe = np.where(sum1==0, precision, sum1)        # 防止除 0
    part = (norm1 / sum1_safe) ** 2                        # (N, M)
    h1 = (1 - part.sum(axis=0)) / (1 - 1/numDays)         # (M,)

    mids  = (maxs + mins)/2
    m     = (1 - spans/(mids.clip(min=precision))).clip(min=0)

    norm2 = (absR / maxs.clip(min=precision))**2
    sum2 = norm2.sum(axis=0)
    sum2_safe = np.where(sum2==0, precision, sum2)
    part2 = (norm2 / sum2_safe) ** 2
    h2 = (1 - part2.sum(axis=0)) / (1 - 1/numDays)

    dist_scores = (h1 + h2 + m + zscr)/4 * isNotST_mask

    pos = np.where(R>0,  R, 0)
    neg = np.where(R<=0, -R, 0)

    numPos = (R>0).sum(axis=0)
    numNeg = (R<=0).sum(axis=0)
    score21 = (numPos - numNeg)/numDays

    sumPos = pos.sum(axis=0); sumNeg = neg.sum(axis=0)
    avgPos = sumPos/np.where(numPos>0, numPos, 1)
    avgNeg = sumNeg/np.where(numNeg>0, numNeg, 1)
    score22 = (avgPos - avgNeg)

    score23 = ((RO>0).sum(axis=0) - (RO<=0).sum(axis=0))/numDays

    pos2 = (pos**2).sum(axis=0); neg2 = (neg**2).sum(axis=0)
    score24 = (pos2 - neg2) / np.where(pos2+neg2>0, pos2+neg2, precision)
    score24 = np.clip(score24, -1, 1)

    posneg_scores = (score21 + score22 + score23 + score24)/4 * isNotST_mask

    ztb_scores   = Z.sum(axis=0)/numDays * isNotST_mask
    dtb_scores   = (1 - D.sum(axis=0)/numDays) * isNotST_mask
    cum_scores   = R.sum(axis=0)
    score_cpos   = np.where(cum_scores>0,  (1 - cum_scores), 0)
    score_cneg   = np.where(cum_scores<=0, (1 - np.abs(cum_scores)), 0)
    cumsum_scores= ((score_cpos + score_cneg)/2) * isNotST_mask

    featScores = {
            'returns_{abs-count-below}_c': level_scores,
            'returns_{abs-dist}_c':        dist_scores,
            'returns_{pos-vs-neg}_c':      posneg_scores,
            'returns_{%ZTB}_c':            ztb_scores,
            'returns_{1-%DTB}_c':          dtb_scores,
            'returns_{cumsum}_c':          cumsum_scores,
        }
    featRanks = {
            k: pd.Series(v, index=symbols).rank(ascending=False, method='first')
            for k, v in featScores.items()
    }
    featScores = {k: pd.Series(v, index=symbols) for k, v in featScores.items()}

    return featRanks, featScores

def _RankSymbolsByTradingVolume(asOfDate, numDays=10):
    idx   = date2idx[asOfDate]
    start = idx - numDays + 1
    win_vol = volume_arr[start:idx+1, :]   # (N, S)
    win_ret = returns_arr[start:idx+1, :]
    mask    = isNotST_arr[start:idx+1, :].all(axis=0)  # (S,)


    n, S = win_vol.shape
    all_nan = np.isnan(win_vol).all(axis=0)
    mask_ser    = pd.Series(mask,    index=symbols)
    all_nan_ser = pd.Series(all_nan, index=symbols)

    # ———— (1) Direction 分档 ————
    win_argmax = np.where(np.isnan(win_vol), -np.inf, win_vol)
    win_argmin = np.where(np.isnan(win_vol),  np.inf, win_vol)
    argmax = win_argmax.argmax(axis=0)
    argmin = win_argmin.argmin(axis=0)
    a_up   = (argmax + 1) / n
    a_dn   = (argmin + 1) / n
    a      = a_up - a_dn
    one_minus_dn = 1 - a_dn
    h2     = (win_vol[1:] > win_vol[:-1]).sum(axis=0) / (n-1)
    cum    = np.nan_to_num(win_vol, nan=0.0).cumsum(axis=0); total = cum[-1]
    head_tail_count = np.zeros(S)
    for i in range(1, n):
        head = cum[i-1] / i
        tail = (total - cum[i-1]) / (n-i)
        head_tail_count += (tail >= head)
    h1     = head_tail_count / (n-1)
    dirc_scores = pd.Series((a + a_up + one_minus_dn + h1 + h2)/5,
                             index=symbols)
    dirc_scores = dirc_scores.where(~all_nan_ser)
    dirc_scores = dirc_scores.where(mask_ser, 0)

    # 2) distribution
    mins = np.nanmin(np.where(np.isnan(win_vol), np.inf, win_vol), axis=0)
    maxs = np.nanmax(np.where(np.isnan(win_vol), -np.inf, win_vol), axis=0)
    mins[all_nan] = np.nan
    maxs[all_nan] = np.nan
    spans = np.maximum(maxs - mins, precision)

    norm1 = ((win_vol - mins) / spans) ** 2
    sum1  = norm1.sum(axis=0)
    with np.errstate(invalid='ignore'):
        h1_d = 1 - np.sum((norm1 / sum1) ** 2, axis=0)
    h1_d = h1_d / (1 - 1 / n)

    mids      = (maxs + mins) / 2
    m         = np.clip(1 - spans / np.where(mids == 0, precision, mids), a_min=0, a_max=None)

    maxs_safe = np.maximum(maxs, precision)
    norm2     = (win_vol / maxs_safe) ** 2
    sum2      = norm2.sum(axis=0)
    with np.errstate(invalid='ignore'):
        h2_d = 1 - np.sum((norm2 / sum2) ** 2, axis=0)
    h2_d = h2_d / (1 - 1 / n)

    mu    = win_vol.mean(axis=0)
    sigma = win_vol.std(axis=0)
    sigma = np.where(sigma == 0, precision, sigma)
    z     = transferZScore(mu / sigma)

    dist_scores = pd.Series((h1_d + h2_d + m + z) / 4,
                             index=symbols)
    dist_scores = dist_scores.where(~all_nan_ser)
    dist_scores = dist_scores.where(mask_ser, 0)

    # 3) pos-vs-neg
    pos = np.where(win_ret > 0, win_vol, np.nan)
    neg = np.where(win_ret <= 0, win_vol, np.nan)
    pos_sum  = np.nansum(pos, axis=0)
    neg_sum  = np.nansum(neg, axis=0)
    pos_mean = np.divide(pos_sum,
                         (win_ret > 0).sum(axis=0),
                         out=np.full_like(pos_sum, np.nan),
                         where=(win_ret > 0).sum(axis=0) > 0)
    neg_mean = np.divide(neg_sum,
                         (win_ret <= 0).sum(axis=0),
                         out=np.full_like(neg_sum, np.nan),
                         where=(win_ret <= 0).sum(axis=0) > 0)
    sum_pm = pos_mean + neg_mean
    mean_score = np.divide(pos_mean - neg_mean, sum_pm,
                           out=np.full_like(sum_pm, np.nan),
                           where=sum_pm != 0)
    sum_score  = np.divide(pos_sum - neg_sum, pos_sum + neg_sum,
                           out=np.full_like(pos_sum, np.nan),
                           where=(pos_sum + neg_sum) != 0)

    posneg_scores = pd.Series((mean_score + sum_score) / 2,
                               index=symbols)
    posneg_scores = posneg_scores.where(~all_nan_ser)
    posneg_scores = posneg_scores.where(mask_ser, 0)

    featScores = {
      'trdv_{dirc}_c':       dirc_scores,
      'trdv_{dist}_c':       dist_scores,
      'trdv_{pos-vs-neg}_c': posneg_scores
    }
    featRanks = {k: v.rank(ascending=False, method='first')
                 for k,v in featScores.items()}
    return featRanks, featScores

def _RankSymbolsByTradingAmount(dictBySymbol, asOfDate, numDays=10):
    amount_df = amount_panel.loc[util.GetLastEquityTradingDates(asOfDate, numDays)]
    mask = isNotST_panel.loc[amount_df.index].all(axis=0)

    levels = [1e8, 3e8, 5e8, 8e8, 1e9]
    level_df = pd.concat([
        (amount_df > lvl).sum(axis=0).div(numDays).rename(f"lvl_{int(lvl/1e8)}")
        for lvl in levels
    ], axis=1).mul(mask, axis=0)
    level_scores = level_df.mean(axis=1)
    level_ranks  = level_scores.rank(ascending=False, method='first')

    mu    = amount_df.mean(axis=0)
    sigma = amount_df.std(axis=0).replace(0, util.precisionError)
    z     = transferZScore(mu.div(sigma))

    mins  = amount_df.min(axis=0)
    maxs  = amount_df.max(axis=0)
    spans = (maxs - mins).replace(0, util.precisionError)

    norm1 = ((amount_df - mins) / spans) ** 2
    sum1  = norm1.sum(axis=0)
    h1    = (1 - (norm1.div(sum1, axis=1)**2).sum(axis=0)) / (1 - 1/numDays)

    mids      = (maxs + mins) / 2
    m         = (1 - spans.div(mids.replace(0, util.precisionError))).clip(lower=0)

    maxs_safe = maxs.replace(0, util.precisionError)
    norm2     = (amount_df.div(maxs_safe, axis=1)) ** 2
    sum2      = norm2.sum(axis=0)
    h2        = (1 - (norm2.div(sum2, axis=1)**2).sum(axis=0)) / (1 - 1/numDays)

    dist_df     = pd.DataFrame({'h1': h1, 'h2': h2, 'm': m, 'z': z})
    dist_df    = dist_df.mul(mask, axis=0)
    dist_scores = dist_df.mean(axis=1)
    dist_ranks  = dist_scores.rank(ascending=False, method='first')

    featScores = {
        'amount_{level}_c': level_scores,
        'amount_{dist}_c':  dist_scores
    }
    featRanks = {
        'amount_{level}_c': level_ranks,
        'amount_{dist}_c':  dist_ranks
    }

    return featRanks, featScores

def _RankSymbolsByAmountCV(dictBySymbol, asOfDate, numMonths=3):
    """Rank symbols by coefficient of variation of trading amount.

    CV = mean(amount) / std(amount) over the past ``numMonths`` months
    (approximated by 21 trading days per month).
    """
    numDays = numMonths * 21
    window = util.GetLastEquityTradingDates(asOfDate, numDays)
    amount_df = amount_panel.loc[window]
    mask = isNotST_panel.loc[window].all(axis=0)

    mu = amount_df.mean(axis=0)
    sigma = amount_df.std(axis=0).replace(0, util.precisionError)
    cv = mu.div(sigma) * mask

    scores = pd.Series(cv, index=symbols)
    ranks = scores.rank(ascending=False, method='first')

    featScores = {'amount_{cv}_c': scores}
    featRanks = {'amount_{cv}_c': ranks}
    return featRanks, featScores

def _RankSymbolsByKBarFeature(dictBySymbol,asOfDate, numDays=10):
    idx   = date2idx[asOfDate]
    start = idx - numDays + 1
    hs    = high_arr [start:idx+1, :]  # (N, S)
    ls    = low_arr  [start:idx+1, :]
    ops   = open_arr [start:idx+1, :]
    cs    = close_arr[start:idx+1, :]
    valid = st_mask  [start:idx+1, :].all(axis=0)  # (S,)

    seq_h = (hs[1:] > hs[:-1]).mean(axis=0)
    seq_l = (ls[1:] > ls[:-1]).mean(axis=0)
    seq_c = (cs[1:] > cs[:-1]).mean(axis=0)
    fs0   = ((seq_h + seq_l + seq_c) / 3) * valid

    def dir_score(arr):
        n, S = arr.shape
        argmax = arr.argmax(axis=0)
        argmin = arr.argmin(axis=0)
        a_up = (argmax + 1) / n
        a_dn = (argmin + 1) / n
        h2   = (arr[1:] > arr[:-1]).mean(axis=0)
        cum  = arr.cumsum(axis=0)
        total = cum[-1]
        i     = np.arange(1, n)[:, None]
        head  = cum[:-1] / i
        tail  = (total - cum[:-1]) / (n - i)
        h1    = (tail >= head).sum(axis=0) / (n - 1)
        return ((a_up - a_dn) + a_up + (1 - a_dn) + h1 + h2) / 5

    fs1 = dir_score(hs) * valid
    fs2 = dir_score(ls) * valid
    fs3 = dir_score(cs) * valid

    hlc   = hs + ls - ops - cs
    hl_oc = (hlc > 0).mean(axis=0)
    oc    = ((cs - ops) > 0).mean(axis=0)
    fs4   = ((hl_oc + oc) / 2) * valid

    featScores = {
        'kBar_{dirc}_c':   pd.Series(fs0, index=symbols),
        'kBar_{h-dirc}_c': pd.Series(fs1, index=symbols),
        'kBar_{l-dirc}_c': pd.Series(fs2, index=symbols),
        'kBar_{c-dirc}_c': pd.Series(fs3, index=symbols),
        'kBar_{ohlc}_c':   pd.Series(fs4, index=symbols),
    }
    featRanks = {
        k: v.rank(ascending=False, method='first')
        for k, v in featScores.items()
    }
    return featRanks, featScores

def _RankSymbolsByMarketValue(dictByDate, asOfDate):
    print(f"=== Ranking by Market Value on {asOfDate} ===")

    df = dictByDate[asOfDate]
    mv = df['marketValue']
    isNotST = df['isNotST'].astype(int)

    absScore = 1.0 - np.floor(mv / _marketValueStep) / 100.0
    absScore = absScore.clip(lower=0)

    bins = pd.cut(
        mv,
        bins=__marketValueBreakpoints,
        labels=False,
        include_lowest=True
    )
    rltvScore = (1.0 - bins * 0.05).clip(lower=0)

    aggScore = ((absScore + rltvScore) / 2.0) * isNotST

    ranks = aggScore.rank(ascending=False, method='first')

    featScores = {'mv_{level}_c': aggScore}
    featRanks  = {'mv_{level}_c': ranks}
    return featRanks, featScores

def _RankSymbolsByDDX(dictBySymbol, asOfDate, numDays=10):
    # 1) 窗口日期
    window = util.GetLastEquityTradingDates(asOfDate, numDays)

    # 2) 切片取 N×S DDX 数据 和 非ST 掩码
    ddx_df = ddx_panel.loc[window]
    mask   = non_st_panel.loc[window].all(axis=0)
    mask_ser = pd.Series(mask, index=ddx_df.columns)

    # 3) Group1: 方向性相关评分
    arr     = ddx_df.values                        # ndarray (N, S)
    gt      = (arr >  0).sum(axis=0)
    le      = (arr <= 0).sum(axis=0)
    score11 = (gt - le) / numDays

    n, S    = arr.shape
    all_nan = np.isnan(arr).all(axis=0)
    all_nan_ser = pd.Series(all_nan, index=ddx_df.columns)
    # avoid nanargmax/argmin on all-NaN slices by replacing NaN with ±inf
    arr_argmax = np.where(np.isnan(arr), -np.inf, arr)
    arr_argmin = np.where(np.isnan(arr),  np.inf, arr)
    idx_max = arr_argmax.argmax(axis=0)
    idx_min = arr_argmin.argmin(axis=0)
    aUp     = (idx_max + 1) / n
    aDn     = (idx_min + 1) / n

    grp1 = pd.DataFrame({
        'score11': score11,
        'score12': aUp - aDn,
        'score13': aUp,
        'score14': 1 - aDn
    }, index=ddx_df.columns)

    group1_scores = grp1.mean(axis=1)
    group1_scores = group1_scores.where(~all_nan_ser)
    group1_scores = group1_scores.where(mask_ser, 0)
    group1_ranks  = group1_scores.rank(ascending=False, method='first')

    # 4) Group2: 正负强度评分
    pos_sum = np.where(arr >  0,  arr, 0).sum(axis=0)
    neg_sum = np.where(arr <= 0, -arr, 0).sum(axis=0)
    denom   = pos_sum + neg_sum
    score21 = (pos_sum - neg_sum) / np.where(denom == 0, util.precisionError, denom)

    pos_max = np.where(arr >  0,  arr, 0).max(axis=0)
    neg_min = np.where(arr <= 0, -arr, 0).min(axis=0)
    score22 = (pos_max - neg_min > 0).astype(int)

    grp2 = pd.DataFrame({
        'score21': score21,
        'score22': score22
    }, index=ddx_df.columns)

    group2_scores = grp2.mean(axis=1)
    group2_scores = group2_scores.where(~all_nan_ser)
    group2_scores = group2_scores.where(mask_ser, 0)
    group2_ranks  = group2_scores.rank(ascending=False, method='first')

    # 5) 返回字典
    featScores = {
        'ddx_{dirc}_c':       group1_scores,
        'ddx_{pos-vs-neg}_b': group2_scores
    }
    featRanks = {
        'ddx_{dirc}_c':       group1_ranks,
        'ddx_{pos-vs-neg}_b': group2_ranks
    }
    return featRanks, featScores

def _RankSymbolsByTechnicalFeature(dictBySymbol, asOfDate, numDays=10):

    dates = util.GetLastEquityTradingDates(asOfDate, numDays)
    # 1. 切片 N×S 的窗口，纯 C 级操作，毫秒级
    macd_df      = feature_panels['MACD'     ].loc[dates]
    crx_df       = feature_panels['Cross_MACD'].loc[dates]
    dea_df       = feature_panels['DEA'      ].loc[dates]
    dif_df       = feature_panels['DIF'      ].loc[dates]
    slope_dea_df = feature_panels['slope_DEA'].loc[dates]
    slope_dif_df = feature_panels['slope_DIF'].loc[dates]

    # 2. isNotST 也提前做好板，按列 all()
    mask = isNotST_panel.loc[dates].all(axis=0)

    # 3. 计算子分
    score1 = (macd_df > 0).sum(axis=0) / numDays
    n_crx   = crx_df.replace(0, np.nan).count(axis=0)
    sum_crx = crx_df.sum(axis=0)
    score2  = sum_crx.div(n_crx)
    score3  = ((dea_df > 0) & (dif_df > 0)).sum(axis=0) / numDays
    score4  = ((slope_dea_df > 0) & (slope_dif_df > 0)).sum(axis=0) / numDays

    # 4. 汇总 + 排名
    scores      = pd.DataFrame({'s1': score1, 's2': score2, 's3': score3, 's4': score4})
    final       = scores.mean(axis=1).mul(mask)
    final_ranks = final.rank(ascending=False, method='first')

    return {'tec_{macd}_c': final_ranks}, {'tec_{macd}_c': final}

def _RankSymbolsByShadowRatios(dictBySymbol, asOfDate, numDays=10):
    idx   = date2idx[asOfDate]
    start = idx - numDays + 1
    window = slice(start, idx+1)

    H = high_panel.iloc[window]
    L = low_panel.iloc[window]
    O = open_panel.iloc[window]
    C = close_panel.iloc[window]

    span = (H - L).replace(0, np.nan)
    oc_max = np.maximum(O.values, C.values)
    oc_min = np.minimum(O.values, C.values)
    OCmax  = pd.DataFrame(oc_max, index=H.index, columns=H.columns)
    OCmin  = pd.DataFrame(oc_min, index=H.index, columns=H.columns)

    upper = (H - OCmax).clip(lower=0).div(span).clip(0,1).mask(span.isna())
    lower = (OCmin - L).clip(lower=0).div(span).clip(0,1).mask(span.isna())

    upper_ratio = upper.mean(axis=0)
    lower_ratio = lower.mean(axis=0)

    # 4) 用 isNotST_panel 做非 ST 过滤
    mask = isNotST_panel.iloc[window].all(axis=0).astype(int)
    upper_ratio *= mask
    lower_ratio *= mask

    # 5) 排名
    featScores = {
        'shadow_upper_ratio': upper_ratio,
        'shadow_lower_ratio': lower_ratio
    }
    featRanks = {
        'shadow_upper_ratio': upper_ratio.rank(ascending=False, method='first'),
        'shadow_lower_ratio': lower_ratio.rank(ascending=False, method='first')
    }
    return featRanks, featScores

def _RankSymbolsByPriceVolumeDivergence(dictBySymbol, asOfDate, numDays=10):
    """Rank symbols by the divergence between price and volume.

    The divergence is measured as the negative Pearson correlation between
    VWAP and volume within the lookback window. Larger positive scores imply
    stronger price-volume divergence.
    """
    idx   = date2idx[asOfDate]
    start = idx - numDays + 1
    win_vwap = vwap_panel.iloc[start:idx+1]
    win_vol  = volume_panel.iloc[start:idx+1]
    mask = isNotST_panel.iloc[start:idx+1].all(axis=0).astype(int)
    corr = win_vwap.corrwith(win_vol, axis=0).fillna(0.0)
    scores = (-corr).mul(mask)
    ranks  = scores.rank(ascending=False, method='first')
    featScores = {'pv_divergence_c': scores}
    featRanks  = {'pv_divergence_c': ranks}
    return featRanks, featScores


def winsorize_df(df: pd.DataFrame, limits: float = 0.01) -> pd.DataFrame:
    """Clip extreme values of a DataFrame cross-sectionally.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by dates with symbols as columns.
    limits : float, optional
        Quantile used for clipping on both tails, by default 0.01.
    """
    lower = df.quantile(limits, axis=1)
    upper = df.quantile(1 - limits, axis=1)
    return df.clip(lower=lower, upper=upper, axis=0)


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardize a DataFrame cross-sectionally."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def neutralize_df(df: pd.DataFrame, neutralizer: pd.DataFrame) -> pd.DataFrame:
    """Neutralize factor values against a neutralizer via cross-sectional regression.

    Parameters
    ----------
    df : pd.DataFrame
        Factor values, indexed by dates with symbols as columns.
    neutralizer : pd.DataFrame
        Exposure values used for neutralization (e.g., log market value) with
        the same shape as ``df``.
    """

    # Ensure numeric dtype before running regressions to avoid casting errors
    df = df.apply(pd.to_numeric, errors="coerce")
    neutralizer = neutralizer.apply(pd.to_numeric, errors="coerce")

    for dt in df.index:
        y = df.loc[dt]
        x = neutralizer.loc[dt]
        mask = y.notna() & x.notna()
        if mask.sum() < 2:
            continue
        X = np.column_stack(
            [np.ones(mask.sum()), x[mask].to_numpy(dtype=float)]
        )
        beta = np.linalg.lstsq(
            X, y[mask].to_numpy(dtype=float), rcond=None
        )[0]
        resid = y[mask] - X @ beta
        df.loc[dt, mask[mask].index] = resid
    return df

LoadFromTmp = True
DoFirstThreeSteps = False
DumpToTmp = False
if LoadFromTmp:
    with open('/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp/dates.pickle', 'rb') as f:
            dates = pickle.load(f)
    with open('/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp/dictByDate.pickle', 'rb') as f:
            dictByDate = pickle.load(f)
    with open('/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp/dictBySymbol.pickle', 'rb') as f:
            dictBySymbol = pickle.load(f)
    print("< LoadFromTmp done >")
symbols = list(dictBySymbol.keys())
if DoFirstThreeSteps:
    GenerateNonSTLabel(dictByDate, dictBySymbol)
    GenerateReturns(symbols, dictBySymbol, dictByDate)
    GenerateFeaturesBySymbol(symbols, dictBySymbol)

if DumpToTmp:
    with open('/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp/dates.pickle', 'wb') as f:
        pickle.dump(dates, f, pickle.HIGHEST_PROTOCOL)
    with open('/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp/dictByDate.pickle', 'wb') as f:
        pickle.dump(dictByDate, f, pickle.HIGHEST_PROTOCOL)
    with open('/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp/dictBySymbol.pickle', 'wb') as f:
        pickle.dump(dictBySymbol, f, pickle.HIGHEST_PROTOCOL)
    print('< Dump done >')

for symbol, df in dictBySymbol.items():
    df['isNotST'] = (df['st'] > 0) & (~df['amount'].isna())
### Initiator Bin
util.set_trading_dates(dates)
symbols = list(dictBySymbol.keys())
dates_idx = pd.Index(dates) 

ma_windows = [5, 10, 20, 34, 55, 233]


isNotST_mask = np.array([int(df['isNotST'].all()) for _, df in dictBySymbol.items()]) 
date2idx = {d: i for i, d in enumerate(dates)}
score_keys = {
    'level':      'to_{dirc}_c',
    'dist':       'to_{dist}_c',
    'pos_vs_neg': 'to_{pos-vs-neg}_c'
}
feature_cols = ['MACD', 'Cross_MACD', 'DEA', 'DIF', 'slope_DEA', 'slope_DIF']
precision = util.precisionError
all_dates = util._global_trading_dates

FACTOR_SWITCHES = {
    'trend': True,
    'to_{dirc}_c': True,
    'to_{dist}_c': True,
    'to_{pos-vs-neg}_c': True,
    'actv_{%ZTB}_c': False,
    'returns_{abs-count-below}_c': True,
    'trdv_{dirc}_c': True,
    'trdv_{dist}_c': True,
    'trdv_{pos-vs-neg}_c': True,
    'amount_{level}_c': True,
    'amount_{dist}_c': True,
    #成交额变异系数#
    'amount_{cv}_c': True,
    'kBar_{dirc}_c': True,
    'kBar_{h-dirc}_c': True,
    'kBar_{l-dirc}_c': True,
    'kBar_{c-dirc}_c': True,
    'kBar_{ohlc}_c': True,
    'mv_{level}_c': True,
    'ddx_{dirc}_c': True,
    'ddx_{pos-vs-neg}_b': True,
    'tec_{macd}_c': True,
    'shadow_{upper}_c': True,
    'shadow_{lower}_c': True,
    #市值中性化换手率残差#
    'to_{mv_resid}_c': True,
    #成交量价背离度#
    'pv_divergence_c':True,
    #月度相对换手率溢出#
    'to_{rel_turnover}_c': True
}


#panelize
returns_oc_panel = pd.DataFrame({sym: df['returns_oc'] for sym, df in dictBySymbol.items()}, index=dates_idx)
isZTB_panel  = pd.DataFrame({sym: df['isZTB'].astype(int)  for sym, df in dictBySymbol.items()},index=dates_idx)
isDTB_panel  = pd.DataFrame({sym: df['isDTB'].astype(int)  for sym, df in dictBySymbol.items()},index=dates_idx)
isNotST_panel  = pd.DataFrame({sym: df['isNotST'].astype(int) for sym, df in dictBySymbol.items()}, index=all_dates)
amount_panel = pd.DataFrame({sym: df['amount'] for sym, df in dictBySymbol.items()}, index=dates_idx)
turnover_panel = pd.DataFrame({sym: df['turnover_adj'] for sym, df in dictBySymbol.items()},index=dates_idx)
returns_panel    = pd.DataFrame({sym: df['returns']    for sym, df in dictBySymbol.items()}, index=dates_idx)
high_panel   = pd.DataFrame({sym: df['high']    for sym, df in dictBySymbol.items()}, index=dates_idx)
low_panel    = pd.DataFrame({sym: df['low']     for sym, df in dictBySymbol.items()}, index=dates_idx)
open_panel   = pd.DataFrame({sym: df['open']    for sym, df in dictBySymbol.items()}, index=dates_idx)
close_panel  = pd.DataFrame({sym: df['close']   for sym, df in dictBySymbol.items()}, index=dates_idx)
st_panel     = pd.DataFrame({sym: df['isNotST'] for sym, df in dictBySymbol.items()}, index=dates_idx)
volume_panel   = pd.DataFrame({sym: df['volume'] for sym,df in dictBySymbol.items()}, index=dates_idx)
vwap_panel     = amount_panel.div(volume_panel.replace(0, np.nan))
marketValue_panel = pd.DataFrame({sym: df['marketValue'] for sym, df in dictBySymbol.items()}, index=dates_idx)
log_mv_panel = np.log(marketValue_panel.replace(0, np.nan))
isNotST_panel  = pd.DataFrame({sym: df['isNotST'].astype(int) for sym,df in dictBySymbol.items()}, index=dates_idx)
ddx_panel = pd.DataFrame({sym: df['ddx'] for sym, df in dictBySymbol.items()}, index=dates_idx)
non_st_panel = pd.DataFrame({sym: df['st'] > 0 for sym, df in dictBySymbol.items()}, index=dates) & pd.DataFrame({sym: df['amount'].notna() for sym, df in dictBySymbol.items()}, index=dates_idx)
feature_panels = {col: pd.DataFrame({sym: df[col] for sym, df in dictBySymbol.items()}, index=all_dates)
    for col in feature_cols
}
isNotST_array = isNotST_panel.values 
ma_dfs    = {w: close_panel.rolling(w).mean() for w in ma_windows}

slope_dfs = {w: ma_dfs[w].diff() for w in ma_windows}

volume_arr  = volume_panel.values  # (D, S)
returns_arr = returns_panel.values
isNotST_arr = isNotST_panel.values
symbols     = volume_panel.columns.tolist()

high_arr   = high_panel.to_numpy()    # shape (D, S)
low_arr    = low_panel.to_numpy()
open_arr   = open_panel.to_numpy()
close_arr  = close_panel.to_numpy()
st_mask    = st_panel.to_numpy().astype(bool)  # shape (D, S)

#mv
__marketValueBreakpoints = [
    0, 5e9, 1e10, 1.5e10, 2e10, 2.5e10, 3e10, 3.5e10, 4e10, 
    4.5e10, 5e10, 5.5e10, 6e10, 6.5e10, 7e10, 7.5e10, 8e10,
    8.5e10, 9e10, 9.5e10, 1e11, 1e13]
_marketValueStep = 2e9

###Initiator bin ended

DoScore = True
if DoScore:
    #####################################
    startDate = datetime.date(2015, 1, 1)
    endDate = datetime.date(2023, 12, 31)
    ######################################
    datesToTest = list(filter(lambda x: x>=startDate and x <= endDate,  dates))
    datesToPickle = datesToTest
    #datesToPickle = [d for d in datesToTest if util.IsLastEquityTradingDateInWeek(d, datesToTest)]

N = 5
__dstDir = '/Users/zym/Desktop/tmp_file_transfer/bar-daily-2024-3-31/tmp'

if FACTOR_SWITCHES['trend']:
    trend_score_df = pd.DataFrame({
        dt: _RankSymbolsByCloseTrend(dictBySymbol, dt, N)[1]['trnd_score']
        for dt in datesToPickle
    }).T.reindex(index=datesToPickle, columns=symbols)
else:
    trend_score_df = None

turnover_keys = ('to_{dirc}_c', 'to_{dist}_c', 'to_{pos-vs-neg}_c')
turnover_score_df = {}
if any(FACTOR_SWITCHES[k] for k in turnover_keys):
    turnover_score_df = {k: pd.DataFrame(index=datesToPickle, columns=symbols)
                         for k in turnover_keys if FACTOR_SWITCHES[k]}
    for dt in datesToPickle:
        scores = _RankSymbolsByTurnover(dictBySymbol, dt, N)[1]
        for k, df in turnover_score_df.items():
            df.loc[dt] = scores[k]

if FACTOR_SWITCHES['actv_{%ZTB}_c']:
    actv_score_df = pd.DataFrame({
        dt: _RankSymbolsByActiveness(dictBySymbol, dt, N)[1]['actv_{%ZTB}_c']
        for dt in datesToPickle
    }).T.reindex(index=datesToPickle, columns=symbols)
else:
    actv_score_df = None

if FACTOR_SWITCHES['returns_{abs-count-below}_c']:
    returns_score_df = pd.DataFrame({
        dt: _RankSymbolsByReturns(dictBySymbol, dt, N)[1]['returns_{abs-count-below}_c']
        for dt in datesToPickle
    }).T.reindex(index=datesToPickle, columns=symbols)
else:
    returns_score_df = None


if FACTOR_SWITCHES['to_{mv_resid}_c']:
    to_mv_resid_score_df = np.log(
        turnover_panel.rolling(window=20, min_periods=20).mean().replace(0, np.nan)
    ).reindex(index=datesToPickle, columns=symbols)
    to_mv_resid_score_df = to_mv_resid_score_df.where(
        isNotST_panel.reindex(index=datesToPickle, columns=symbols).astype(bool)
    )
else:
    to_mv_resid_score_df = None

if FACTOR_SWITCHES['to_{rel_turnover}_c']:
    short_avg = turnover_panel.rolling(window=20, min_periods=20).mean()
    long_avg  = turnover_panel.rolling(window=250, min_periods=250).mean()
    rel_to_df = short_avg.div(long_avg).replace([np.inf, -np.inf], np.nan)
    rel_to_score_df = rel_to_df.reindex(index=datesToPickle, columns=symbols)
    rel_to_score_df = rel_to_score_df.where(
        isNotST_panel.reindex(index=datesToPickle, columns=symbols).astype(bool)
    )
else:
    rel_to_score_df = None

trdv_keys = ('trdv_{dirc}_c', 'trdv_{dist}_c', 'trdv_{pos-vs-neg}_c')
trdv_score_df = {}
if any(FACTOR_SWITCHES[k] for k in trdv_keys):
    trdv_score_df = {k: pd.DataFrame(index=datesToPickle, columns=symbols)
                     for k in trdv_keys if FACTOR_SWITCHES[k]}
    for dt in datesToPickle:
        scores = _RankSymbolsByTradingVolume(dt, N)[1]
        for k, df in trdv_score_df.items():
            df.loc[dt] = scores[k]

# --- Price-Volume Divergence scores ---
if FACTOR_SWITCHES['pv_divergence_c']:
    pv_div_score_df = pd.DataFrame(index=datesToPickle, columns=symbols)
    for dt in datesToPickle:
        scores = _RankSymbolsByPriceVolumeDivergence(dictBySymbol, dt, N)[1]
        pv_div_score_df.loc[dt] = scores['pv_divergence_c']

# --- Trading amount scores ---
amount_score_df = pd.DataFrame()
amount_rel_score_df = pd.DataFrame()
amount_cv_score_df = pd.DataFrame()
if any(FACTOR_SWITCHES[k] for k in ('amount_{level}_c', 'amount_{dist}_c')):
    amount_score_df = pd.DataFrame(index=datesToPickle, columns=symbols)
    amount_rel_score_df = pd.DataFrame(index=datesToPickle, columns=symbols)
    for dt in datesToPickle:
        scores = _RankSymbolsByTradingAmount(dictBySymbol, dt, N)[1]
        if FACTOR_SWITCHES['amount_{level}_c']:
            amount_score_df.loc[dt] = scores['amount_{level}_c']
        if FACTOR_SWITCHES['amount_{dist}_c']:
            amount_rel_score_df.loc[dt] = scores['amount_{dist}_c']
if FACTOR_SWITCHES['amount_{cv}_c']:
    amount_cv_score_df = pd.DataFrame(index=datesToPickle, columns=symbols)
    for dt in datesToPickle:
        scores = _RankSymbolsByAmountCV(dictBySymbol, dt, 3)[1]
        amount_cv_score_df.loc[dt] = scores['amount_{cv}_c']
# --- K-Bar features ---
kbar_keys = ('kBar_{dirc}_c', 'kBar_{h-dirc}_c', 'kBar_{l-dirc}_c',
             'kBar_{c-dirc}_c', 'kBar_{ohlc}_c')
kbar_enabled = [k for k in kbar_keys if FACTOR_SWITCHES[k]]
kbar_score_df = {}
if kbar_enabled:
    kbar_score_df = {k: pd.DataFrame(index=datesToPickle, columns=symbols) for k in kbar_enabled}
    for dt in datesToPickle:
        scores = _RankSymbolsByKBarFeature(dictBySymbol, dt, N)[1]
        for k in kbar_enabled:
            kbar_score_df[k].loc[dt] = scores[k]
    kBar_dirc_score_df   = kbar_score_df.get('kBar_{dirc}_c')
    kBar_h_dirc_score_df = kbar_score_df.get('kBar_{h-dirc}_c')
    kBar_l_dirc_score_df = kbar_score_df.get('kBar_{l-dirc}_c')
    kBar_c_dirc_score_df = kbar_score_df.get('kBar_{c-dirc}_c')
    kBar_ohlc_score_df   = kbar_score_df.get('kBar_{ohlc}_c')
else:
    kBar_dirc_score_df = kBar_h_dirc_score_df = kBar_l_dirc_score_df = None
    kBar_c_dirc_score_df = kBar_ohlc_score_df = None

if FACTOR_SWITCHES['mv_{level}_c']:
    mv_score_df = pd.DataFrame({
        dt: _RankSymbolsByMarketValue(dictByDate, dt)[1]['mv_{level}_c']
        for dt in datesToPickle
    }).T.reindex(index=datesToPickle, columns=symbols)
else:
    mv_score_df = None

ddx_keys = ('ddx_{dirc}_c', 'ddx_{pos-vs-neg}_b')
ddx_enabled = [k for k in ddx_keys if FACTOR_SWITCHES[k]]
ddx_score_df = {}
if ddx_enabled:
    ddx_score_df = {k: pd.DataFrame(index=datesToPickle, columns=symbols) for k in ddx_enabled}
    for dt in datesToPickle:
        scores = _RankSymbolsByDDX(dictBySymbol, dt, N)[1]
        for k in ddx_enabled:
            ddx_score_df[k].loc[dt] = scores[k]
    ddx_dirc_score_df = ddx_score_df.get('ddx_{dirc}_c')
    ddx_posneg_score_df = ddx_score_df.get('ddx_{pos-vs-neg}_b')
else:
    ddx_dirc_score_df = ddx_posneg_score_df = None

if FACTOR_SWITCHES['tec_{macd}_c']:
    tech_score_df = pd.DataFrame({
        dt: _RankSymbolsByTechnicalFeature(dictBySymbol, dt, N)[1]['tec_{macd}_c']
        for dt in datesToPickle
    }).T
    tech_score_df = tech_score_df.reindex(index=datesToPickle, columns=symbols)
else:
    tech_score_df = None


# --- Shadow ratios ---
upper_shadow_df = lower_shadow_df = None
shadow_keys = []
if FACTOR_SWITCHES['shadow_{upper}_c']:
    shadow_keys.append('shadow_upper_ratio')
if FACTOR_SWITCHES['shadow_{lower}_c']:
    shadow_keys.append('shadow_lower_ratio')
shadow_score_df = {}
if shadow_keys:
    shadow_score_df = {k: pd.DataFrame(index=datesToPickle, columns=symbols) for k in shadow_keys}
    for dt in datesToPickle:
        scores = _RankSymbolsByShadowRatios(dictBySymbol, dt, N)[1]
        for k in shadow_keys:
            shadow_score_df[k].loc[dt] = scores[k]
    upper_shadow_df = shadow_score_df.get('shadow_upper_ratio')
    lower_shadow_df = shadow_score_df.get('shadow_lower_ratio')

score_dfs = {}
if FACTOR_SWITCHES['trend']:
    score_dfs['trend'] = trend_score_df
if FACTOR_SWITCHES['to_{dirc}_c'] and 'to_{dirc}_c' in turnover_score_df:
    score_dfs['to_{dirc}_c'] = turnover_score_df['to_{dirc}_c']
if FACTOR_SWITCHES['to_{dist}_c'] and 'to_{dist}_c' in turnover_score_df:
    score_dfs['to_{dist}_c'] = turnover_score_df['to_{dist}_c']
if FACTOR_SWITCHES['to_{pos-vs-neg}_c'] and 'to_{pos-vs-neg}_c' in turnover_score_df:
    score_dfs['to_{pos-vs-neg}_c'] = turnover_score_df['to_{pos-vs-neg}_c']
if FACTOR_SWITCHES['actv_{%ZTB}_c']:
    score_dfs['actv_{%ZTB}_c'] = actv_score_df
if FACTOR_SWITCHES['returns_{abs-count-below}_c']:
    score_dfs['returns_{abs-count-below}_c'] = returns_score_df
if FACTOR_SWITCHES['trdv_{dirc}_c'] and 'trdv_{dirc}_c' in trdv_score_df:
    score_dfs['trdv_{dirc}_c'] = trdv_score_df['trdv_{dirc}_c']
if FACTOR_SWITCHES['trdv_{dist}_c'] and 'trdv_{dist}_c' in trdv_score_df:
    score_dfs['trdv_{dist}_c'] = trdv_score_df['trdv_{dist}_c']
if FACTOR_SWITCHES['trdv_{pos-vs-neg}_c'] and 'trdv_{pos-vs-neg}_c' in trdv_score_df:
    score_dfs['trdv_{pos-vs-neg}_c'] = trdv_score_df['trdv_{pos-vs-neg}_c']
if FACTOR_SWITCHES['amount_{level}_c']:
    score_dfs['amount_{level}_c'] = amount_score_df
if FACTOR_SWITCHES['amount_{dist}_c']:
    score_dfs['amount_{dist}_c'] = amount_rel_score_df
if FACTOR_SWITCHES['amount_{cv}_c']:
    score_dfs['amount_{cv}_c'] = amount_cv_score_df
if FACTOR_SWITCHES['kBar_{dirc}_c'] and kBar_dirc_score_df is not None:
    score_dfs['kBar_{dirc}_c'] = kBar_dirc_score_df
if FACTOR_SWITCHES['kBar_{h-dirc}_c'] and kBar_h_dirc_score_df is not None:
    score_dfs['kBar_{h-dirc}_c'] = kBar_h_dirc_score_df
if FACTOR_SWITCHES['kBar_{l-dirc}_c'] and kBar_l_dirc_score_df is not None:
    score_dfs['kBar_{l-dirc}_c'] = kBar_l_dirc_score_df
if FACTOR_SWITCHES['kBar_{c-dirc}_c'] and kBar_c_dirc_score_df is not None:
    score_dfs['kBar_{c-dirc}_c'] = kBar_c_dirc_score_df
if FACTOR_SWITCHES['kBar_{ohlc}_c'] and kBar_ohlc_score_df is not None:
    score_dfs['kBar_{ohlc}_c'] = kBar_ohlc_score_df
if FACTOR_SWITCHES['mv_{level}_c'] and mv_score_df is not None:
    score_dfs['mv_{level}_c'] = mv_score_df
if FACTOR_SWITCHES['ddx_{dirc}_c'] and ddx_dirc_score_df is not None:
    score_dfs['ddx_{dirc}_c'] = ddx_dirc_score_df
if FACTOR_SWITCHES['ddx_{pos-vs-neg}_b'] and ddx_posneg_score_df is not None:
    score_dfs['ddx_{pos-vs-neg}_b'] = ddx_posneg_score_df
if FACTOR_SWITCHES['tec_{macd}_c'] and tech_score_df is not None:
    score_dfs['tec_{macd}_c'] = tech_score_df
if FACTOR_SWITCHES['shadow_{upper}_c'] and upper_shadow_df is not None:
    score_dfs['shadow_{upper}_c'] = upper_shadow_df
if FACTOR_SWITCHES['shadow_{lower}_c'] and lower_shadow_df is not None:
    score_dfs['shadow_{lower}_c'] = lower_shadow_df
if FACTOR_SWITCHES['pv_divergence_c'] and pv_div_score_df is not None:
    score_dfs['pv_divergence_c'] = pv_div_score_df
if FACTOR_SWITCHES['to_{mv_resid}_c'] and to_mv_resid_score_df is not None:
    score_dfs['to_{mv_resid}_c'] = to_mv_resid_score_df
if FACTOR_SWITCHES['to_{rel_turnover}_c'] and rel_to_score_df is not None:
    score_dfs['to_{rel_turnover}_c'] = rel_to_score_df

for name, df in score_dfs.items():
    df = winsorize_df(df)
    df = standardize_df(df)
    df = neutralize_df(df, log_mv_panel)
    score_dfs[name] = df

rank_dfs = {
    name: df.rank(axis=1, ascending=False, method='first')
    for name, df in score_dfs.items()
}

for dt in datesToPickle:
    lstOfScores = {
        name: (
            rank_dfs[name].loc[dt], 
            score_dfs[name].loc[dt]
        )
        for name in score_dfs
    }
    dstFileName = os.path.join(__dstDir, f'score_{N}_{dt}.pickle')
    print(f"Pickling scores on {dt} to {dstFileName}")
    with open(dstFileName, 'wb') as f:
        pickle.dump(lstOfScores, f, pickle.HIGHEST_PROTOCOL)



print("Step 2 Complete")