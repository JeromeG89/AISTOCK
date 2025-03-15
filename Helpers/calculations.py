import numpy as np

def Cal(data):
    data['Dividends'] = data['Dividends']/data['Adj Close']
    df = data.copy()
    #Volatility
    df['volatility'] = df['Adj Close'].rolling(window = 12, min_periods = 12).std()
    

    #BOV
    dof = np.sign(df['Adj Close'] - df['Adj Close'].shift(1))
    Vol_Change = (df['Volume'] * dof)
    bov = Vol_Change.cumsum()
    df['BOV'] = bov.pct_change()


    Low_60 = df['Adj Close'].rolling(window = 60, min_periods = 60).min()
    High_60 = df['Adj Close'].rolling(window = 60, min_periods = 60).max()
    Dif = High_60 - Low_60
    def Low_cal(k):
        arr = Low_60 + (Dif * k)
        Parr = (df['Adj Close']/arr) -1
        return Parr
    def High_cal(j):
        arr = High_60 - (Dif * j)
        Parr = (df['Adj Close']/arr) -1
        return Parr
    
    #Fibonnaci
    df['SF2'] = Low_cal(0.5)
    df['SF4'] = Low_cal(0.786)
    df['RC1'] = High_cal(0.786)
    
    #Boolinger
    df['SMA'] = df['Adj Close'].rolling(window = 20, min_periods = 20).mean()
    df['STD'] = df['Adj Close'].rolling(window = 20, min_periods = 20).std()
    df['LBol'] = df['SMA'] - (2* df['STD'])
    df['HBol'] = df['SMA'] + (2* df['STD'])
    df['PLbol'] = ((df['Adj Close'] - df['LBol']) / df['LBol']) 
    df['PHbol'] = ((df['Adj Close'] - df['HBol']) / df['HBol'])
    df.drop(['LBol','HBol','STD','SMA'], inplace = True, axis = 1)
    high_prices = df['High']
    close_prices = df['Adj Close']
    low_prices = df['Low']

    nine_period_high =  df['High'].rolling(window=9).max()
    nine_period_low = df['Low'].rolling(window=9).min()
    tenkan_sen = (nine_period_high + nine_period_low) /2

    period26_high = high_prices.rolling(window=26).max()
    period26_low = low_prices.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    period52_high = high_prices.rolling(window=52).max()
    period52_low = low_prices.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    df['kumo'] = senkou_span_a - senkou_span_b
    df['kijun_tenkan'] = tenkan_sen - kijun_sen
    ema12 = df['Adj Close'].ewm(span = 12,adjust=False, min_periods = 12).mean()
    ema26 = df['Adj Close'].ewm(span = 26,adjust=False, min_periods = 26).mean()
    MACD = ema12 - ema26
    df['MACD'] = MACD
    ema9 = df['MACD'].ewm(span = 9,adjust=False, min_periods = 9).mean()
    df['ema9'] = ema9
    df['MACD_Histogram'] = MACD - ema9
    df['MACD_grad'] = df['MACD'] - df['MACD'].shift(1)
    df['ema9_grad'] = df['ema9'] - df['ema9'].shift(1)
    cup = df['Adj Close'].diff().copy()
    cdown = df['Adj Close'].diff().copy()
    cup[cup < 0] = 0
    cdown[cdown > 0] = 0
    cup_sum = cup.rolling(14, min_periods = 14).sum()
    cdown_sum = cdown.rolling(14, min_periods = 14).sum() 
    rsi_cal = cup_sum / (cdown_sum * -1)
    RSI = 100 - (100/(1 + rsi_cal))
    df['RSI'] = RSI
    def wwma(values, n):
        return values.ewm(alpha=1/n, adjust=False).mean()

    def atr(df, n=14):
        data_work = df.copy()
        high = data_work['High']
        low = data_work['Low']
        close = data_work['Adj Close']
        data_work['tr0'] = abs(high - low)
        data_work['tr1'] = abs(high - close.shift())
        data_work['tr2'] = abs(low - close.shift())
        tr = data_work[['tr0', 'tr1', 'tr2']].max(axis=1)
        atr = wwma(tr, n)
        return atr
    df['atr'] = atr(df, 14)
    
    def parabolic_sar(df, step=0.02, max_af=0.2):

        length = len(df)
        if length == 0:
            return df

        # Prepare lists to store intermediate values
        psar = [0.0] * length        # Parabolic SAR values
        ep = [0.0] * length          # Extreme points
        af = [0.0] * length          # Acceleration factors
        uptrend = [True] * length    # Trend direction flags (True for uptrend, False for downtrend)

        if length > 1:
            first_high, first_low = df['High'].iloc[0], df['Low'].iloc[0]
            second_high, second_low = df['High'].iloc[1], df['Low'].iloc[1]
            if second_high < first_high and second_low < first_low:
                uptrend[0] = False  # start with downtrend if second bar is lower on both High and Low
        # Set initial SAR and EP based on identified initial trend
        if uptrend[0]:  
            psar[0] = df['Low'].iloc[0]    # initial SAR is first period's Low (bullish start)
            ep[0] = df['High'].iloc[0]    # initial EP is first period's High
        else:
            psar[0] = df['High'].iloc[0]   # initial SAR is first period's High (bearish start)
            ep[0] = df['Low'].iloc[0]     # initial EP is first period's Low
        af[0] = step  # start with initial acceleration factor (e.g., 0.02)

        # ----- Iterate over each period to compute PSAR -----
        for i in range(1, length):
            prev_index = i - 1
            prev_psar = psar[prev_index]
            prev_ep = ep[prev_index]
            prev_af = af[prev_index]
            prev_uptrend = uptrend[prev_index]

            if prev_uptrend:  # If the previous trend was bullish
                # Calculate tentative SAR for current period (uptrend formula)
                curr_sar = prev_psar + prev_af * (prev_ep - prev_psar)
                # SAR cannot exceed the lowest Low of the previous two periods
                if i >= 2:
                    curr_sar = min(curr_sar, df['Low'].iloc[i-1], df['Low'].iloc[i-2])
                else:
                    curr_sar = min(curr_sar, df['Low'].iloc[i-1])
                # Check if uptrend continues or reverses
                if df['Low'].iloc[i] < prev_psar:
                    # *Reversal to downtrend*: price fell below prior SAR
                    uptrend[i] = False
                    psar[i] = prev_ep            # set SAR to previous High (EP from uptrend)
                    ep[i] = df['Low'].iloc[prev_index]   # reset EP to previous period's Low
                    af[i] = step                 # reset AF to initial step (0.02)
                else:
                    # *Continue uptrend*
                    uptrend[i] = True
                    psar[i] = curr_sar
                    # Update EP and AF if a new High is made
                    if df['High'].iloc[i] > prev_ep:
                        ep[i] = df['High'].iloc[i]
                        af[i] = min(max_af, prev_af + step)  # increase AF up to max_af
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:  # Previous trend was bearish
                # Calculate tentative SAR for current period (downtrend formula)
                curr_sar = prev_psar - prev_af * (prev_psar - prev_ep)
                # SAR cannot go below the highest High of the previous two periods
                if i >= 2:
                    curr_sar = max(curr_sar, df['High'].iloc[i-1], df['High'].iloc[i-2])
                else:
                    curr_sar = max(curr_sar, df['High'].iloc[i-1])
                # Check if downtrend continues or reverses
                if df['High'].iloc[i] > prev_psar:
                    # *Reversal to uptrend*: price rose above prior SAR
                    uptrend[i] = True
                    psar[i] = prev_ep             # set SAR to previous Low (EP from downtrend)
                    ep[i] = df['High'].iloc[prev_index]   # reset EP to previous period's High
                    af[i] = step                  # reset AF to initial step (0.02)
                else:
                    # *Continue downtrend*
                    uptrend[i] = False
                    psar[i] = curr_sar
                    # Update EP and AF if a new Low is made
                    if df['Low'].iloc[i] < prev_ep:
                        ep[i] = df['Low'].iloc[i]
                        af[i] = min(max_af, prev_af + step)  # increase AF up to max_af
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
        df['psar'] = psar / df['Adj Close']
    parabolic_sar(df)
    
    return df


if (__name__ == "__main__"):
    import yfinance as yf
    import pandas as pd
    stock = yf.Ticker("NVDA")
    df = Cal(stock.history(period = '31mo', interval ='1wk', auto_adjust = False))

    print(df.loc[df.index[2]]['High'])
    print(df[['High', 'Low', 'Adj Close', 'psar']])
    