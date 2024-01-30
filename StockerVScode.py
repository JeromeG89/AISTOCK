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
from sklearn.metrics import make_scorer, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
st =  time.time()

sp500_tickers = ["MMM", "ABT", "ABBV", "ABMD", "ACN", "ADBE", "AMD", "AAP", "AES",
    "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE",
    "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP",
    "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI",
    "ANSS",  "AON", "AOS", "APA", "AAPL", "AMAT", "APTV", "ADM", "ANET",
    "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "BKR",
    "BAC", "BK", "BAX", "BDX", "BBY", "BIO", "BIIB", "BLK", "BA",
    "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "CHRW",
    "CDNS", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT",
    "CBOE", "CBRE", "CDW", "CE", "CNC", "CNP", "CF", "SCHW", "CHTR",
    "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG",
    "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG",  "COP",
    "ED", "STZ", "COO", "CPRT", "GLW", "CTVA", "COST", "CCI", "CSX", "CMI", "CVS",
    "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", "DVN", "DXCM", "FANG", "DLR",
    "DFS",  "DISH", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE",
    "DUK", "DD", "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA",
    "EMR", "ENPH", "ETR", "EOG", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EVRG",
    "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FAST", "FRT",
    "FDX", "FIS", "FITB", "FE",  "FLT",  "FLS", "FMC", "F",
    "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GPS", "GRMN", "IT", "GNRC",
    "GD", "GE", "GIS", "GM", "GPC", "GILD", "GL", "GPN", "GS", "GWW", "HAL", "HBI",
    "HIG", "HAS", "HCA", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUM",
    "HBAN", "HII", "IEX", "IDXX", "ITW", "ILMN", "INCY", "IR", "INTC",
    "ICE", "IBM", "IP", "IPG", "IFF", "INTU", "ISRG", "IVZ", "IPGP", "IQV", "IRM",
    "JKHY", "J", "JBHT", "SJM", "JNJ", "JCI", "JPM", "JNPR", "K", "KEY",
    "KEYS", "KMB", "KIM", "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX",
    "LW", "LVS", "LEG", "LDOS", "LEN", "LLY", "LNC", "LIN", "LYV", "LKQ", "LMT",
    "L", "LOW", "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM",
    "MAS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM",
    "MCHP", "MU", "MSFT", "MAA", "MHK", "TAP", "MDLZ", "MPWR", "MNST", "MCO",
    "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA",
    "NWS", "NEE", "NKE", "NI", "NSC", "NTRS", "NOC", "NCLH",
    "NOV", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC",
    "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PH", "PAYX", "PAYC", "PYPL", "PENN",
    "PNR", "PEP", "PRGO", "PFE", "PM", "PSX", "PNW", "PXD", "PNC",
    "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PTC", "PEG", "PSA",
    "PHM", "PVH", "QRVO", "PWR", "QCOM","DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RHI", "ROK",
"ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE",
"NOW", "SHW", "SPG", "SWKS", "SNA", "SO", "LUV", "SWK", "SBUX", "STT", "STE",
"SYK", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TGT", "TEL",
"TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG",
"TRV", "TRMB", "TFC", "TYL", "TSN", "UDR", "ULTA", "USB", "UAA", "UA",
"UNP", "UAL", "UNH", "UPS", "URI", "UHS", "UNM", "VLO", "VTR", "VRSN",
"VRSK", "VZ", "VRTX", "V", "VNO", "VMC", "WRB", "WAB", "WMT", "WBA",
"DIS", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WU", "WRK", "WY",
"WHR", "WMB",  "WYNN", "XEL", "XRX", "XYL", "YUM", "ZBRA",
"ZBH", "ZION", "ZTS"
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
     'EXC', 'FIS', 'VRSK', 'ROK',
    'RSG', 'ODFL', 'PPG', 'EA', 'KMI',
    'CSGP', 'GPN', 'CMI', 'MRNA', 'XEL',
    'KDP', 'IR', 'ON', 'DD', 'VICI',
    'CTVA', 'MLM', 'BKR', 'EXR', 'FICO',
    'ED', 'CDW', 'KR', 'EFX', 'VMC',
    'DG', 'HAL'
]

ticker_list2 =    [    
    "ICE", "MO", "CDNS", "CME", "SHW", 
    "ITW", "BDX", "SLB", "CSX", "USB",
    
    "EOG", "NOC", "TGT", "CL", "PYPL", 
    "WM", "MCK", "ANET", "PNC", "CMG"
]
ticker_list_Fav = ['PG', 'MO', 'EXC', 'AMAT', 'T', 'GOOGL', 'BAC', 'WFC', 'MA', 'GOOG', 'META', 'CAT', 'AMGN', 'TJX', 'MS', 'BKNG', 'PGR', 'ETN', 'NET', 'HD']

ticker_list3 = ["pypl", 'LLY', "NVDA", "TGT", "FANG"]
for tick in ['AIG']:
    Ticker = tick
    data_new = yf.Ticker(Ticker).history(period = '31mo', interval ='1wk', auto_adjust = False)
    data_old = yf.Ticker(Ticker).history(period = '62mo', interval ='1wk', auto_adjust = False)
    luist3 = []

    for K in range(min_num,4):
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
        df.drop(['Open','High','Low', 'Volume'], axis = 1, inplace = True)
        sp500_daily_change.index = df.index
        df["Feels_goodline"] = sp500_daily_change
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
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        param_grid = {
            'classifier__max_depth': [None] + list(np.linspace(3,15,5,dtype=int)),
            'classifier__min_samples_split': [2, 5, 10, 15],
            'classifier__min_samples_leaf': list(np.linspace(3,15,5,dtype=int)),
            'classifier__max_features': ['sqrt', 'log2', None],  # Adjust based on the number of features
            'classifier__criterion': ['gini', 'entropy']
        }
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=30000))
        ])
        param_grid = {
            'scaler__with_mean': [True, False],
            'scaler__with_std': [True, False],
            'classifier__penalty': ['l2'],
            'classifier__C': np.logspace(-3, 3, 7)
        }"""
        luist_2 = []
        luist4 = []
        custom_scorer = make_scorer(f1_score, zero_division=1)
        for i in list(range(5,16,5)):
            grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(n_splits= i), scoring=custom_scorer, n_jobs = cores, error_score=np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                grid_search.fit(X_train, y_train)
            if (len(np.unique(np.array(grid_search.predict(X_train)))) == 2):  
                luist_2.append([grid_search.score(X_train,y_train)*100, grid_search.score(X_val,y_val)*100, grid_search.score(X_test,y_test)*100, grid_search.predict(z), list(enumerate(np.unique(grid_search.predict(X_train), return_counts = True)[1]))])
            else:
                luist_2.append([grid_search.score(X_train,y_train)*100,grid_search.score(X_val,y_val)*100, grid_search.score(X_test,y_test)*100,np.nan, list(enumerate(np.unique(grid_search.predict(X_train), return_counts = True)[1]))])
        for i in range(len(luist_2)):
            luist4.append(luist_2[i][1])
        luist3.append(luist_2[luist4.index(max(luist4))])
        """    
        max_value = max(luist_2, key=luist_2.get)
        abc = GridSearchCV(pipeline, param_grid, cv=KFold(n_splits= max_value), scoring='accuracy')
        abc.fit(X_train,y_train)
        print(len(y_train), len(y_val), len(y_test))
        if (len(np.unique(np.array(abc.predict(X_train)))) == 2):  
            luist3.append([abc.score(X_train,y_train)*100, abc.score(X_val,y_val)*100,abc.score(X_test,y_test)*100, abc.predict(z)])
        else:
            luist3.append([abc.score(X_train,y_train)*100,abc.score(X_test,y_test)*100,np.nan])"""
    for i in range(len(luist3)):
        print(tick, ":","For K =", i + min_num , "prediction output is:", luist3[i][3], "Train/Validation/Test Confidence: ", round(luist3[i][0],0), " / ",round(luist3[i][1],0)," / ",round(luist3[i][2],0), "Current prices are:", list(df_new_p.iloc[-(i+ min_num):]['Adj Close']), "Shape:", list(input_shapes), luist3[i][4])  
print(list(df_new_p.iloc[-K:]['Adj Close'].index.date))
et = time.time()
print("Time elapsed = ", round((et-st)/60,2), "Mins")