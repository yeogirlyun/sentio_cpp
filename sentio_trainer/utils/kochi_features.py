import numpy as np
from typing import List, Tuple


def _clip_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.clip(b, 1e-12, None)


def kochi_feature_names() -> List[str]:
    return [
        # Time features
        "HOUR_SIN", "HOUR_COS", "DOW_SIN", "DOW_COS", "WOY_SIN",
        "WOY_COS",
        # Overlap
        "SMA_5", "SMA_20", "EMA_5", "EMA_20",
        "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
        "TENKAN_SEN", "KIJUN_SEN", "SENKOU_SPAN_A", "CHIKOU_SPAN",
        "SAR",
        "TURBULENCE", "VWAP", "VWAP_ZSCORE",
        # Momentum
        "AROON_UP", "AROON_DOWN", "AROON_OSC",
        "MACD", "MACD_SIGNAL", "MACD_HIST", "MOMENTUM", "ROC",
        "RSI",
        "STOCH_SLOWK", "STOCH_SLOWD", "STOCH_CROSS", "WILLR",
        "PLUS_DI", "MINUS_DI", "ADX", "CCI", "PPO", "ULTOSC",
        "SQUEEZE_ON", "SQUEEZE_OFF",
        # Volatility/volume-flow
        "STDDEV", "ATR", "ADL_30", "OBV_30", "VOLATILITY",
        "KELTNER_UPPER", "KELTNER_LOWER", "KELTNER_MIDDLE",
        "GARMAN_KLASS",
        # Price action / candles
        "LOG_RETURN_1", "OVERNIGHT_GAP", "BAR_SHAPE", "SHADOW_UP",
        "SHADOW_DOWN", "DOJI", "HAMMER", "ENGULFING",
    ]


def compute_kochi_features(df) -> Tuple[List[str], np.ndarray]:
    import importlib
    pd = importlib.import_module("pandas")  # type: ignore[assignment]

    def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
        s = pd.Series(a)
        return s.rolling(window=w, min_periods=1).mean().to_numpy()

    def _rolling_std(a: np.ndarray, w: int) -> np.ndarray:
        s = pd.Series(a)
        return s.rolling(window=w, min_periods=1).std().fillna(0.0).to_numpy()

    def _ema(a: np.ndarray, w: int) -> np.ndarray:
        s = pd.Series(a)
        return s.ewm(span=w, adjust=False).mean().to_numpy()

    # Ensure datetime index (UTC)
    if not hasattr(df.index, "tz"):
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)

    openp = df["open"].astype(np.float64).to_numpy()
    high = df["high"].astype(np.float64).to_numpy()
    low = df["low"].astype(np.float64).to_numpy()
    close = df["close"].astype(np.float64).to_numpy()
    vol = df["volume"].astype(np.float64).to_numpy()
    n = close.shape[0]

    # Time features
    ny = df.index.tz_convert("America/New_York")
    hours = ny.hour.to_numpy()
    hsin = np.sin(hours * (2 * np.pi / 24))
    hcos = np.cos(hours * (2 * np.pi / 24))
    dows = ny.dayofweek.to_numpy()
    dwsin = np.sin(dows * (2 * np.pi / 5))
    dwcos = np.cos(dows * (2 * np.pi / 5))
    woy = ny.isocalendar().week.to_numpy()
    wsin = np.sin(woy * (2 * np.pi / 52))
    wcos = np.cos(woy * (2 * np.pi / 52))

    # SMA/EMA diffs
    sma5 = _rolling_mean(close, 5) - close
    sma20 = _rolling_mean(close, 20) - close
    ema5 = _ema(close, 5) - close
    ema20 = _ema(close, 20) - close

    # Bollinger 20
    m20 = _rolling_mean(close, 20)
    sd20 = _rolling_std(close, 20)
    bb_u = (m20 + 2.0 * sd20) - close
    bb_l = (m20 - 2.0 * sd20) - close
    bb_m = m20 - close

    # Ichimoku simplified
    hh9 = pd.Series(high).rolling(9, min_periods=1).max().to_numpy()
    ll9 = pd.Series(low).rolling(9, min_periods=1).min().to_numpy()
    tenkan = 0.5 * (hh9 + ll9)
    hh26 = pd.Series(high).rolling(26, min_periods=1).max().to_numpy()
    ll26 = pd.Series(low).rolling(26, min_periods=1).min().to_numpy()
    kijun = 0.5 * (hh26 + ll26)
    senkou_a = 0.5 * (tenkan + kijun)
    chikou = np.roll(close, -26) - close
    sar = high - low
    turbul = _clip_div(high - low, np.maximum(openp, 1e-8))

    # VWAP (rolling 14) and zscore over 20
    num = pd.Series(close * vol).rolling(14, min_periods=1).sum().to_numpy()
    den = pd.Series(vol).rolling(14, min_periods=1).sum().to_numpy()
    vwap = _clip_div(num, den) - close
    vmean = _rolling_mean(vwap, 20)
    vstd = _rolling_std(vwap, 20)
    vwap_z = np.divide(vwap - vmean, vstd, out=np.zeros_like(vstd), where=vstd > 0)

    # Aroon 14
    aroon_up = np.zeros(n)
    aroon_dn = np.zeros(n)
    for i in range(n):
        s = max(0, i - 13)
        hh = high[s:i + 1]
        ll = low[s:i + 1]
        idx_up = np.argmax(hh) + s
        idx_dn = np.argmin(ll) + s
        aroon_up[i] = 1.0 - (i - idx_up) / max(1, 14)
        aroon_dn[i] = 1.0 - (i - idx_dn) / max(1, 14)
    aroon_osc = aroon_up - aroon_dn

    # MACD simplified
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macds = macd
    macdh = macd - macds

    # Momentum, ROC, RSI (scaled 0..1)
    prev10 = np.concatenate([np.full(10, close[0]), close[:-10]])
    momentum = close - prev10
    roc = _clip_div(close, np.maximum(prev10, 1e-12)) - 1.0
    gain = np.maximum(close - np.concatenate([[close[0]], close[:-1]]), 0)
    loss = np.maximum(np.concatenate([[close[0]], close[:-1]]) - close, 0)
    rs_gain = pd.Series(gain).rolling(14, min_periods=1).mean().to_numpy()
    rs_loss = pd.Series(loss).rolling(14, min_periods=1).mean().to_numpy()
    rsi = np.where(rs_gain + rs_loss > 0, rs_gain / (rs_gain + rs_loss), 1.0)

    # Stochastics
    hh14 = pd.Series(high).rolling(14, min_periods=1).max().to_numpy()
    ll14 = pd.Series(low).rolling(14, min_periods=1).min().to_numpy()
    stoch_k = np.where(hh14 == ll14, 0.0, (close - ll14) / (hh14 - ll14))
    stoch_d = stoch_k
    stoch_cross = (stoch_k > stoch_d).astype(np.float32)
    willr = np.where(hh14 == ll14, 0.5, (close - ll14) / (hh14 - ll14))

    # Placeholders / simplified
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)
    cci = np.zeros(n)
    ppo = np.zeros(n)
    ultosc = np.zeros(n)

    # Squeeze Madrid approx using BB and Keltner
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr20 = pd.Series(tr).rolling(20, min_periods=1).mean().to_numpy()
    kc_ub = m20 + 1.5 * atr20
    kc_lb = m20 - 1.5 * atr20
    squeeze_on = (
        (m20 - 2 * sd20 > kc_lb) & (m20 + 2 * sd20 < kc_ub)
    ).astype(np.float32)
    squeeze_off = (
        (m20 - 2 * sd20 < kc_lb) & (m20 + 2 * sd20 > kc_ub)
    ).astype(np.float32)

    # Volatility set
    stddev = _rolling_std(close, 20)
    atr = atr20
    clv = ((close - low) - (high - close)) / np.clip(high - low, 1e-12, None)
    adl = pd.Series(clv * vol).rolling(30, min_periods=1).sum().to_numpy()
    obv_step = np.where(close > prev_close, vol,
                        np.where(close < prev_close, -vol, 0.0))
    obv = pd.Series(obv_step).rolling(30, min_periods=1).sum().to_numpy()
    ret = np.diff(
        np.log(np.clip(close, 1e-12, None)),
        prepend=np.log(np.clip(close[0], 1e-12, None)),
    )
    var20 = pd.Series(ret).rolling(20, min_periods=1).var().fillna(0.0)
    var20 = var20.to_numpy()
    kcu = (m20 + 2 * atr20) - close
    kcl = (m20 - 2 * atr20) - close
    kcm = m20 - close
    log_hl = np.log(
        np.clip(high / np.clip(low, 1e-12, None), 1e-12, None)
    )
    log_co = np.log(
        np.clip(close / np.clip(openp, 1e-12, None), 1e-12, None)
    )
    gk = (
        0.5 * log_hl * log_hl
        - (2.0 * np.log(2.0) - 1.0) * log_co * log_co
    )
    gk_series = pd.Series(gk)
    gk_mean20 = gk_series.rolling(20, min_periods=1).mean().to_numpy()
    gk = np.sqrt(np.clip(gk_mean20, 0.0, None))

    # Price action
    log_ret1 = np.diff(
        np.log(np.clip(close, 1e-12, None)),
        prepend=np.log(np.clip(close[0], 1e-12, None)),
    )
    overnight_gap = np.zeros(n)
    body = np.abs(close - openp)
    hl = np.clip(high - low, 1e-8, None)
    bar_shape = body / hl
    shadow_up = (high - np.maximum(close, openp)) / hl
    shadow_down = (np.minimum(close, openp) - low) / hl
    doji = np.zeros(n)
    hammer = np.zeros(n)
    engulfing = np.zeros(n)

    cols = kochi_feature_names()
    X = np.column_stack(
        [
            hsin, hcos, dwsin, dwcos, wsin, wcos,
            sma5, sma20, ema5, ema20,
            bb_u, bb_l, bb_m,
            tenkan - close, kijun - close, senkou_a - close, chikou, sar,
            turbul, vwap, vwap_z,
            aroon_up, aroon_dn, aroon_osc,
            macd, macds, macdh, momentum, roc, rsi,
            stoch_k, stoch_d, stoch_cross, willr,
            plus_di, minus_di, adx, cci, ppo, ultosc,
            squeeze_on, squeeze_off,
            stddev, atr, adl, obv, var20,
            kcu, kcl, kcm, gk,
            log_ret1, overnight_gap, bar_shape, shadow_up, shadow_down,
            doji, hammer, engulfing,
        ]
    ).astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return cols, X


