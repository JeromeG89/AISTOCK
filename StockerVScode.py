import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import yfinance as yf
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier
import warnings
st =  time.time()

ticker_list_SG = ['BN4.SI', 'A17U.SI', 'C38U.SI', 'C09.SI', 'D05.SI', 'G13.SI', 'H78.SI', 'J36.SI', 'BN4.SI', 
                  'ME8U.SI', 'M44U.SI', 'S58.SI', 'U96.SI', 'C6L.SI','Z74.SI', 'S68.SI', 'S63.SI', 'Y92.SI', 
                  'U11.SI', 'U14.SI', 'V03.SI', 'F34.SI', 'BS6.SI', 'BUOU.SI', 'EMI.SI', 'S51.SI']
sp500_tickers = [ 'MSFT', 'AAPL', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 'LLY', 'TSLA', 
                 'JPM', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'HD', 'MRK', 'COST', 'ABBV', 'ADBE', 'AMD', 
                 'CRM', 'CVX', 'NFLX', 'WMT', 'KO', 'PEP', 'ACN', 'BAC', 'MCD', 'TMO', 'CSCO', 'LIN', 'ABT',
                 'ORCL', 'CMCSA', 'INTC', 'INTU', 'WFC', 'DIS', 'VZ', 'AMGN', 'IBM', 'DHR', 'CAT', 'NOW',
                 'QCOM', 'PFE', 'UNP', 'GE', 'SPGI', 'TXN', 'PM', 'AMAT', 'UBER', 'ISRG', 'RTX', 'COP', 
                 'HON', 'T', 'LOW', 'GS', 'BKNG', 'NKE', 'PLD', 'NEE', 'BA', 'AXP', 'MDT', 'ELV', 'SYK', 
                 'TJX', 'LRCX', 'MS', 'BLK', 'VRTX', 'ETN', 'PANW', 'C', 'SBUX', 'PGR', 'DE', 'MDLZ', 'UPS', 
                 'ADP', 'REGN', 'CB', 'BMY', 'GILD', 'ADI', 'MMC', 'MU', 'CI', 'BSX', 'LMT', 'CVS', 'SCHW', 
                 'AMT', 'BX', 'FI', 'ZTS', 'SNPS', 'TMUS', 'KLAC', 'CDNS', 'EQIX', 'SO', 'CME', 'DUK', 'ICE', 
                 'MO', 'SHW', 'CSX', 'ITW', 'SLB', 'CL', 'WM', 'BDX', 'ANET', 'CMG', 'PYPL', 'TGT', 'MCK', 'PH', 
                 'EOG', 'PSX', 'ABNB', 'USB', 'NOC', 'TT', 'MPC', 'MCO', 'TDG', 'ORLY', 'APH', 'HCA', 'MAR', 'GD', 
                 'PNC', 'AON', 'ROP', 'FCX', 'APD', 'NSC', 'FDX', 'NXPI', 'ADSK', 'MSI', 'PCAR', 'CTAS', 'EMR', 'GM', 
                 'LULU', 'PXD', 'MMM', 'EW', 'COF', 'ECL', 'AJG', 'HLT', 'AZO', 'TRV', 'TFC', 'AIG', 'WELL', 'ROST', 'F', 
                 'CARR', 'CCI', 'MSCI', 'VLO', 'DXCM', 'HUM', 'MCHP', 'NUE', 'O', 'SPG', 'SRE', 'PSA', 'TEL', 'URI', 'DHI', 
                 'IDXX', 'DLR', 'CPRT', 'GWW', 'BK', 'WMB', 'FTNT', 'CEG', 'AEP', 'KMB', 'ALL', 'SYY', 'MNST', 'AFL', 'MET', 
                 'STZ', 'HES', 'FAST', 'CNC', 'OKE', 'NEM', 'AMP', 'COR', 'LHX', 'PAYX', 'CTSH', 'A', 'IQV', 'AME', 'LEN', 'D',
                 'GIS', 'OXY', 'DOW', 'CTVA', 'PRU', 'OTIS', 'JCI', 'FIS', 'IT', 'ODFL', 'YUM', 'KVUE', 'VRSK', 'GPN', 'RSG', 
                 'BIIB', 'PCG', 'CMI', 'EXC', 'CSGP', 'IR', 'EA', 'PPG', 'XEL', 'KMI', 'MRNA', 'MLM', 'CHTR', 'KDP', 'VICI', 
                 'ED', 'CDW', 'VMC', 'EL', 'HAL', 'FICO', 'ACGL', 'ROK', 'EFX', 'ON', 'MPWR', 'EXR', 'KR', 'KHC', 'DG', 'PWR',
                 'ADM', 'HSY', 'FTV', 'ANSS', 'RCL', 'BKR', 'PEG', 'DLTR', 'GEHC', 'WST', 'RMD', 'KEYS', 'XYL', 'HIG', 'FANG', 
                 'DD', 'DFS', 'DVN', 'ZBH', 'TTWO', 'MTD', 'CBRE', 'WTW', 'DAL', 'EIX', 'CAH', 'WEC', 'TSCO', 'HPQ', 'ULTA', 
                 'GLW', 'AVB', 'TROW', 'CHD', 'SBAC', 'WAB', 'WY', 'AWK', 'BR', 'APTV', 'LYB', 'NVR', 'FITB', 'PHM', 'ILMN', 
                 'WBD', 'STT', 'HWM', 'BLDR', 'DOV', 'MTB', 'STE', 'EBAY', 'DTE', 'FLT', 'ETR', 'PTC', 'RJF', 'IFF', 'MOH',
                 'EQR', 'TDY', 'IRM', 'EXPE', 'DRI', 'GPC', 'HPE', 'BAX', 'ALGN', 'CLX', 'ES', 'CBOE', 'NDAQ', 'TRGP', 'PPL', 
                 'INVH', 'FE', 'WAT', 'HUBB', 'ARE', 'LH', 'AKAM', 'BALL', 'WDC', 'COO', 'VTR', 'LVS', 'FDS', 'CTRA', 'GRMN',
                 'BRO', 'NTAP', 'STLD', 'AXON', 'LUV', 'EXPD', 'HBAN', 'AEE', 'TYL', 'OMC', 'HOLX', 'VRSN', 'CNP', 'CINF',
                 'J', 'PFG', 'JBHT', 'RF', 'MKC', 'ATO', 'STX', 'CMS', 'VLTO', 'TXT', 'JBL', 'NTRS', 'EPAM', 'IEX', 'CCL', 
                 'WRB', 'EG', 'WBA', 'SWKS', 'TSN', 'SYF', 'AVY', 'SNA', 'MAS', 'LW', 'FSLR', 'ESS', 'LDOS', 'CFG', 'MAA', 
                 'TER', 'BBY', 'DPZ', 'CE', 'CF', 'K', 'PKG', 'POOL', 'EQT', 'VTRS', 'CAG', 'SWK', 'DGX', 'ENPH', 'NDSN',
                 'HST', 'SJM', 'AMCR', 'PODD', 'UAL', 'KEY', 'ALB', 'L', 'MRO', 'BG', 'TRMB', 'RVTY', 'IPG', 'LKQ', 'ZBRA',
                 'LYV', 'NRG', 'ROL', 'KIM', 'LNT', 'MGM', 'PNR', 'JKHY', 'GEN', 'JNPR', 'EVRG', 'IP', 'TFX', 'KMX', 'TAP', 'AES',
                 'ALLE', 'CRL', 'UDR', 'FFIV', 'DAY', 'INCY', 'HII', 'TECH', 'NI', 'GL', 'CPT', 'REG', 'MTCH', 'QRVO', 'MOS', 
                 'PEAK', 'HSIC', 'UHS', 'BBWI', 'WRK', 'CTLT', 'EMN', 'AOS', 'AAL', 'PAYC', 'NWSA', 'WYNN', 'APA', 'CZR', 'TPR', 'ETSY',
                 'BXP', 'HRL', 'CPB', 'AIZ', 'CHRW', 'RHI', 'MKTX', 'BWA', 'FOXA', 'FMC', 'PNW', 'BEN', 'FRT', 'NCLH', 'XRAY', 'IVZ', 'GNRC',
                 'PARA', 'CMA', 'BIO', 'HAS', 'WHR', 'DVA', 'ZION', 'RL', 'MHK', 'VFC', 'FOX', 'NWS'
]

sp500 = yf.download(sp500_tickers,period = '62mo', interval = "1wk", )
sp500.drop(["Open", "High", "Low", "Close", "Volume"], axis = 1, inplace = True)
sp500_pct = sp500.pct_change().fillna(0)
sp500_pct[sp500_pct > 0] = 1
sp500_pct[sp500_pct < 0] = -1
sp500_daily_change = sp500_pct.mean(axis = 1)



cores = 3
min_num = 2
ticker_list =   [ 
 'TJX', 'NET', 'HD', 'BKNG', 'AMAT', 'UNH', 'ETN', 'META', 'GOOG', 'ABBV', 'CAT', 'PG', 'GOOGL', 'ANET', 'CME', 'ICE', 'EQIX', 'TMUS', 'ZTS', 'MMC', 'REGN', 'LRCX', 'UBER', 'AMZN', 'ORLY', 'TDG', 'PH', 'MAR', 'CTAS', 'AIG', 'AFL', 'GE', 'CDNS', 'CMG', 'HCA', 'AZO', 'PRU', 'VLO'
]


ticker_list_gudearns = ['TJX', 'UNH', 'BAC', 'CME', 'MMC', 'TDG', 'K', 'UNH', 'V', 'MA', 'JNJ', 'MRK', 'ADBE', 'CRM', 'KO', 'BAC', 'CMCSA', 'VZ', 'NOW', 'PFE', 'NEE', 'TJX', 'C', 'MDLZ', 'ADP', 'MMC', 'SNPS']
ticker_list_Fav = ['MS', 'AMAT', 'PGR', 'ETN', 'PG', 'CAT', 'T', 'UNH', 'ABBV', 'GOOGL', 'BAC', 'GOOG', 
                   'ANET', 'CME', 'ICE', 'EQIX', 'TMUS', 'ZTS', 'MMC', 'REGN', 'LRCX', 'UBER', 'AMZN', 'ORLY', 'TDG', 'PH', 'MAR', 'CTAS', 'AIG', 'AFL', 'GE',
                   'CDNS', 'CMG', 'HCA', 'AZO', 'PRU', 'VLO', 'EA', 'FICO', 'ED', 'CDW', 'EFX', 'VMC', 'AMD', 'ALL', 'AMT', 'K']

ticker_list3 = [ 'MDLZ', 'SNPS', 'CDW', 'VMC']
for tick in ['VMC']:
    stock = yf.Ticker(tick)
    luist3 = []
    for K in range(min_num,5):
        data_new = stock.history(period = '31mo', interval ='1wk', auto_adjust = False)
        data_old = stock.history(period = '62mo', interval ='1wk', auto_adjust = False)
        def Prep(df, old):
            df = df.drop('Close', axis = 1, inplace = False)
            df['Dividends'] = df['Dividends']/df['Adj Close']
            df['dir'] = (df['Adj Close'].shift(-K) - df['Adj Close'])
            if old == True:
                df = df.loc[~df.index.isin(data_new.index)]
            df.loc[df['dir'] > 0, 'dir'] = 1
            df.loc[df['dir'] < 0, 'dir'] = 0
            if old == False:
                df['dir'] = df['dir'].fillna('insf')
            return df
        df_new_p = Prep(data_new, old = False).copy()
        df_old_p = Prep(data_old, old = True).copy()

        def Cal(data3):
            data3['Dividends'] = data3['Dividends']/data3['Adj Close']
            df = data3.copy()
            #Volatility
            df['volatility'] = df['Adj Close'].rolling(window = 12, min_periods = 12).std()
            
            df['dir'] = (df['Adj Close'].shift(-K) - df['Adj Close']) / df['Adj Close']
            #BOV
            dof = np.sign(df['Adj Close'] - df['Adj Close'].shift(1))
            Vol_Change = (df['Volume'] * dof)
            bov = Vol_Change.cumsum()
            df['BOV'] = bov.pct_change()

            df.loc[df['dir'] > 0, 'dir'] = 1
            df.loc[df['dir'] < 0, 'dir'] = 0
            df['dir'] = df['dir'].fillna('insf')
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
        
        df = pd.concat([Cal(df_old_p), Cal(df_new_p)])
        len_dif = len(sp500_daily_change.index) - len(df.index)
        if len_dif > 0:
            newp500_daily_change = sp500_daily_change.drop(index = sp500_daily_change.index[:len_dif], axis = 0, inplace = False)
            newp500_daily_change.index = df.index
            df["Feels_goodline"] = newp500_daily_change
        else:
            sp500_daily_change.index = df.index
            df["Feels_goodline"] = sp500_daily_change
        df.drop(['Open','High','Low', 'Volume'], axis = 1, inplace = True)
        z = df.loc[((df['dir'].loc[df['dir'] == 'insf']).index)]
        z.drop(['dir','Adj Close'], inplace= True, axis = 1)
        df.dropna(inplace = True)
        df.drop(index = z.index, inplace = True, axis = 0)
        z = z[K:]
        y = df['dir']
        y = y.astype('int')
        X = df.drop(['dir','Adj Close'], axis = 1)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.222, shuffle = False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.285, shuffle = False)
        input_shapes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        pipeline = Pipeline([
             ('classifier', RandomForestClassifier(random_state=42))
        ])
        param_grid = {
            'classifier__n_estimators': [30, 50, 100],
            'classifier__max_depth': [3 ,5, 7],
            'classifier__min_samples_split': [2, 5, 10, 15],
            'classifier__min_samples_leaf': [ 3, 7, 10, 15],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__bootstrap': [True],
            'classifier__criterion': ['gini', 'entropy'],
  
        }
        luist_2 = []
        luist4 = []
        custom_scorer = make_scorer(f1_score, zero_division=1)
        for i in list([5]):
            grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(n_splits= i), scoring=custom_scorer, n_jobs = cores, error_score=np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                grid_search.fit(X_train, y_train)
            tn, fp, fn, tp = confusion_matrix(y_test,grid_search.predict(X_test)).ravel()
            tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val,grid_search.predict(X_val)).ravel()
            if tp == 0 or tp_val == 0:
                tp_rate = 0
            else:
                tp_rate = ((tp + tp_val)/(fp + fp_val + tp + tp_val)) *100
            if tn == 0 or tn_val == 0:
                tn_rate = 0
            else:    
                tn_rate = ((tn + tn_val)/(fn + fn_val + tn + tn_val)) * 100
            print([tn, fp, fn, tp])
            print([tn_val, fp_val, fn_val, tp_val])
            if (len(np.unique(np.array(grid_search.predict(X_train)))) == 2):  
                luist_2.append([grid_search.score(X_train,y_train)*100, grid_search.score(X_val,y_val)*100, grid_search.score(X_test,y_test)*100, grid_search.predict(z), list(enumerate([round(tn_rate, 1), round(tp_rate, 1)])), [tn + tn_val, fn + fn_val, tp + tp_val, fp + fp_val]])
            else:
                luist_2.append([grid_search.score(X_train,y_train)*100,grid_search.score(X_val,y_val)*100, grid_search.score(X_test,y_test)*100,np.nan, list(enumerate([round(tn_rate, 1),  round(tp_rate, 1)])),  [tn + tn_val, fn + fn_val, tp + tp_val, fp + fp_val ]])
        for i in range(len(luist_2)):
            luist4.append(luist_2[i][1])
        luist3.append(luist_2[luist4.index(max(luist4))])
    for i in range(len(luist3)):
        print(tick, ":","For K =", i + min_num , "prediction output is:", luist3[i][3], "Train/Validation/Test Confidence: ", round(luist3[i][0],0), " / ",round(luist3[i][1],0)," / ",round(luist3[i][2],0), "Current prices are:", list(round(df_new_p.iloc[-(i+ min_num):]['Adj Close'], 3)), "Shape:", list(input_shapes), luist3[i][4], luist3[i][5])  
print(list(df_new_p.iloc[-K:]['Adj Close'].index.date))
et = time.time()
print("Time elapsed = ", round((et-st)/60,2), "Mins")
