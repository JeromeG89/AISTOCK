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
        
        return df