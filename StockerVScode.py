import pandas as pd
import numpy as np
from intoSQL import toMySQL 
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
import yfinance as yf
import time
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tpot import TPOTClassifier
from sklearn.pipeline import make_pipeline
import warnings
from dateutil.relativedelta import relativedelta
from ManualEarnings import getEarnings
from Collinearity import remove_HighCorrelation

st =  time.time()
import gc
VersionNo = '18'

ticker_list_SG = ['BN4.SI', 'A17U.SI', 'C38U.SI', 'C09.SI', 'D05.SI', 'G13.SI', 'H78.SI', 'J36.SI', 'BN4.SI', 
                  'ME8U.SI', 'M44U.SI', 'S58.SI', 'U96.SI', 'C6L.SI','Z74.SI', 'S68.SI', 'S63.SI', 'Y92.SI', 
                  'U11.SI', 'U14.SI', 'V03.SI', 'F34.SI', 'BS6.SI', 'BUOU.SI', 'EMI.SI', 'S51.SI']
sp500_tickers = ['AAPL','NVDA','MSFT','AMZN','GOOG','GOOGL','META','TSLA','AVGO','WMT','LLY','JPM','V','ORCL',
                 'UNH','XOM','MA','COST','HD','PG','NFLX','JNJ','BAC','CRM','ABBV','TMUS','CVX','KO','MRK','WFC',
                 'ADBE','CSCO','NOW','BX','ACN','AMD','IBM','PEP','MCD','AXP','LIN','DIS','MS','PM','TMO','ABT',
                 'ISRG','CAT','GE','GS','INTU','VZ','QCOM','TXN','BKNG','PLTR','T','DHR','CMCSA','BLK','SPGI',
                 'RTX','NEE','LOW','SCHW','PGR','HON','SYK','ETN','AMGN','PFE','KKR','AMAT','TJX','UNP','UBER',
                 'C','ANET','COP','PANW','BSX','ADP','LMT','DE','BMY','VRTX','FI','NKE','BA','GILD','CB','SBUX',
                 'MU','MMC','ADI','MDT','UPS','PLD','LRCX','MO','SHW','AMT','GEV','EQIX','SO','TT','INTC','CTAS',
                 'PYPL','CRWD','MCO','PH','ICE','APH','WM','CMG','ELV','CI','KLAC','DUK','DELL','CME','ABNB',
                 'REGN','CDNS','MDLZ','PNC','MSI','WELL','AON','USB','MAR','HCA','ITW','SNPS','ZTS','CEG','CL',
                 'MCK','FTNT','GD','EMR','ORLY','MMM','TDG','EOG','COF','APD','ECL','CVS','RCL','WMB','SPG','NOC',
                 'FDX','RSG','CSX','ADSK','CARR','DLR','AJG','OKE','BDX','HLT','TFC','KMI','PCAR','TGT','FCX',
                 'CPRT','AFL','TRV','ROP','MET','NSC','GM','BK','PSA','SLB','FICO','GWW','CHTR','SRE','URI','AZO',
                 'JCI','NXPI','AMP','VST','ALL','PSX','AXON','CMI','ROST','PAYX','AEP','DHI','MNST','O','PWR',
                 'FANG','LULU','MPC','MSCI','HWM','AIG','D','COR','NEM','NDAQ','FAST','FIS','TEL','PRU','DFS',
                 'KMB','PEG','OXY','PCG','LHX','KDP','AME','CCI','LEN','EA','KVUE','HES','STZ','ODFL','KR','GLW',
                 'TRGP','CBRE','EW','GRMN','F','DAL','VLO','IR','CTVA','BKR','VRSK','CTSH','A','XEL','IT','OTIS',
                 'SYY','LVS','YUM','EXC','VMC','KHC','GEHC','ACGL','IQV','GIS','EXR','CCL','IDXX','MLM','RMD',
                 'HSY','WAB','IRM','MTB','HPQ','HIG','DD','HUM','NUE','VICI','ROK','RJF','TTWO','EFX','UAL','ED',
                 'EIX','ETR','WTW','AVB','MCHP','CSGP','FITB','LYV','BRO','HPE','TPL','WEC','XYL','EBAY','DXCM',
                 'DECK','ANSS','TSCO','GPN','CAH','KEYS','DOW','PPG','GDDY','STT','EQR','MPWR','CNC','EL','SW',
                 'ON','K','DOV','BR','TROW','NVR','FTV','TYL','NTAP','MTD','SYF','VLTO','CHD','WBD','PHM','VTR',
                 'EQT','AWK','SMCI','HBAN','CPAY','HAL','DTE','LYB','PPL','HUBB','ADM','WDC','AEE','EXPE','CINF',
                 'PTC','CDW','WRB','ROL','RF','SBAC','FE','WST','BIIB','DVN','WAT','IFF','WY','TSN','TDY','ATO',
                 'ES','PKG','LDOS','ERIE','ZBH','NTRS','CBOE','STE','ZBRA','FOXA','FOX','STLD','STX','MKC','FSLR',
                 'CFG','CLX','CNP','INVH','LUV','BLDR','OMC','NRG','CMS','ESS','DRI','ULTA','IP','COO','LH','PFG',
                 'TER','GEN','MAA','BBY','SNA','KEY','L','PODD','VRSN','CTRA','TRMB','JBHT','FDS','ARE','DG','PNR',
                 'HRL','DGX','MAS','IEX','ALGN','NI','NWSA','NWS','GPC','MRNA','HOLX','J','BALL','KIM','MOH','UDR',
                 'EXPD','AVY','BAX','EG','DPZ','LNT','DLTR','CF','TXT','JBL','VTRS','FFIV','DOC','AKAM','AMCR','INCY',
                 'NDSN','TPR','EVRG','RL','POOL','RVTY','BXP','SWKS','EPAM','PAYC','REG','KMX','HST','APTV','DVA',
                 'SWK','CPT','CAG','UHS','CPB','JKHY','TAP','CHRW','SJM','DAY','ALB','ALLE','NCLH','JNPR','SOLV',
                 'TECH','BG','EMN','AIZ','BEN','CTLT','LW','MGM','IPG','GNRC','AOS','PNW','WYNN','LKQ','CRL','FRT',
                 'ENPH','AES','HAS','HSIC','MKTX','GL','TFX','MTCH','MHK','MOS','IVZ','CZR','APA','PARA','CE','WBA',
                 'BWA','HII','FMC','QRVO','AMTM','BRK.B','BF.B']
sp500 = yf.download(sp500_tickers,period = '62mo', interval = "1wk", )
vix = yf.Ticker('^VIX').history(period = '62mo',interval = "1wk",  )
sp500.drop(["Open", "High", "Low", "Close", "Volume"], axis = 1, inplace = True)
sp500_pct = sp500.pct_change().fillna(0)
sp500_pct[sp500_pct > 0] = 1
sp500_pct[sp500_pct < 0] = -1
sp500_daily_change = sp500_pct.mean(axis = 1)

 
cores = 3
min_num = 1
ticker_list =   []
#'CDW', 'VLO','PG','PH','HD','TDG'
#BEST: 'XOM', 'GE' 
failed = []
confu_level= 0.7
confi_level = 0
roc_level = 0.7
min_num = 2
max_num = 3
mp_tut = {0: 'Short', 1: 'Long'}
filepath = r'Logs\12-23-24.txt'
oldest_date = (datetime.today()+ relativedelta(months=-62)).strftime('%Y-%m-%d')

for tick in ['NVDA', 'FI', 'AVGO'] + sp500_tickers:
    try:
        stock = yf.Ticker(tick)
        #analyst ratings
        file_path_sent = r'Analyst_sent.txt'
        analyst_mp = {}
        with open(file_path_sent,'r') as file: 
            for line in file:
                key, value = line.strip().split(',')
                analyst_mp[key] = float(value)
        recos = stock.upgrades_downgrades
        recos = recos[recos.index.strftime('%Y-%m-%d') >= oldest_date].copy()
        recos['To_Grade_weight'] = recos['ToGrade'].map(analyst_mp)
        recos['From_Grade_weight'] = recos['FromGrade'].map(analyst_mp)
        working = recos[['ToGrade', 'FromGrade', 'To_Grade_weight', 'From_Grade_weight']]
        to_grade = working[['ToGrade', 'To_Grade_weight']].drop_duplicates('ToGrade')
        to_grade = to_grade[to_grade['To_Grade_weight'].isna()]
    
        from_grade = working[['FromGrade', 'From_Grade_weight']].drop_duplicates('FromGrade')
        from_grade = from_grade[from_grade['From_Grade_weight'].isna()]

        #Write for ToGrade
        for sentiment in to_grade['ToGrade']:
            new_weight = input(f'Enter relevant weight "{sentiment}": ')
            analyst_mp[sentiment] = new_weight

        #Write for FromGrade
        for sentiment in from_grade['FromGrade']:
            if sentiment in analyst_mp.keys():
                pass
            else:
                new_weight = input(f'Enter relevant weight "{sentiment}": ')
                analyst_mp[sentiment] = new_weight

        recos['To_Grade_weight'] = recos['ToGrade'].map(analyst_mp).astype('float')
        recos['From_Grade_weight'] = recos['FromGrade'].map(analyst_mp).astype('float')

        with open(file_path_sent, 'w') as file:
            for key, value in analyst_mp.items():
                file.write(f"{key},{value}\n")
        recos['Month'] = recos['Firm'].index.strftime("%Y-%m")
        mp_raw, mp_change  = {}, {}
        mp_raw_list = []
        mp_change_list = []
        recos_values = recos.values
        for i in recos.values:
            mp_raw[i[0]] = i[4]
            mp_change[i[0]] = i[4] - i[5]
            mp_raw_list.append(np.mean(list(mp_raw.values())))
            mp_change_list.append(np.mean(list(mp_change.values())))
        recos['mp_raw'] = mp_raw_list
        recos['mp_change'] = mp_change_list
        anal_sent = recos[['Month','mp_raw', 'mp_change']]
        anal_sent = anal_sent.drop_duplicates('Month', keep = 'last').reset_index(drop = True)

        #Earnings scraper
        try:
            earnings = stock.get_earnings_dates(limit = 26)
            earnings.index = earnings.index.strftime('%Y-%m-%d')
            earnings = earnings['Reported EPS'].dropna()
        except:
            earnings = getEarnings(tick)
        
        
        
        dict_list = []
        
        for K in range(min_num,max_num + 1, 2): #iterates through each num of weeks for ticker, from min_num to max_num inclusive, week n with n + K
            data_new = stock.history(period = '31mo', interval ='1wk', auto_adjust = False)
            data_old = stock.history(period = '62mo', interval ='1wk', auto_adjust = False)
            encoder = OneHotEncoder(drop = 'first', sparse_output = False)
            def Prep(df, old):
                df = df.drop('Close', axis = 1, inplace = False)
                df['Dividends'] = df['Dividends']/df['Adj Close']
                if old == True:
                    df = df.loc[~df.index.isin(data_new.index)]
                df['Month'] = df.index.month
                encoded_months = encoder.fit_transform(df[['Month']])
                encoded_months_df = pd.DataFrame(encoded_months, columns= encoder.get_feature_names_out(['Month']))
                encoded_months_df.index = df.index
                df = df.drop(columns = ['Month']).join(encoded_months_df)
                return df
            df_new_p = Prep(data_new, old = False).copy()
            df_old_p = Prep(data_old, old = True).copy()
            
            def Cal(data3):
                data3['Dividends'] = data3['Dividends']/data3['Adj Close']
                df = data3.copy()
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
            df = pd.concat([Cal(df_old_p), Cal(df_new_p)])
            def target_variable(df):
                direc = df['Adj Close'].shift(-K) - df['Adj Close']
                direc = direc.astype(object)
                direc[direc > 0] = 1
                direc[direc < 0] = 0
                direc[direc.isna()] = 'insf'
                return direc
            
            df['dir'] = target_variable(df)    
            df['Month'] = df.index.strftime('%Y-%m')
            df.index = df.index.strftime('%Y-%m-%d')
            index_holder = df.index
            df = df.merge(earnings, how = 'outer', left_index = True, right_index = True)
            df['Reported EPS'] = df['Reported EPS'].ffill()
            df.dropna(subset = ["Adj Close", "Volume"], inplace = True)
            df["Reported EPS"]  = df["Adj Close"] / df["Reported EPS"]
 
            df = pd.merge(df, anal_sent, on = 'Month', how = 'left').drop('Month', axis = 1)
            df.index = index_holder
            df[['mp_raw', 'mp_change']] = df[['mp_raw', 'mp_change']].ffill()

            len_dif = len(sp500_daily_change.index) - len(df.index)
            if len_dif > 0:
                newp500_daily_change = sp500_daily_change.drop(index = sp500_daily_change.index[:len_dif], axis = 0, inplace = False)
                newp500_daily_change.index = df.index
                df["Feels_goodline"] = newp500_daily_change
                
            else:
                sp500_daily_change.index = df.index
                df["Feels_goodline"] = sp500_daily_change

            df['vix'] = vix.iloc[-len(df):]['Close'].values

            #print(df) #For debugging
            
            df.drop(['Open','High','Low', 'Volume', 'Adj Close'], axis = 1, inplace = True)
            df.dropna(inplace = True)

            df_y = df['dir']
            df = remove_HighCorrelation(df.drop(['dir'], axis = 1), 0.7)
            df =  df.join(df_y)

            z = df[df['dir'] == 'insf'].copy()
            z.drop(['dir'], inplace= True, axis = 1)
            
            df.drop(index = z.index, inplace = True, axis = 0)
            y = df['dir']
            y = y.astype('int')
            X = df.drop(['dir'], axis = 1)
            
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , shuffle = False, random_state= 42)
            input_shapes = [X_train.shape[0], X_test.shape[0]]
            tpot_pipeline = make_pipeline(StandardScaler(), TPOTClassifier(generations=10, population_size= 50, verbosity=1, scoring = 'f1', n_jobs= cores, random_state= 42))
            tpot_pipeline.fit(X_train, y_train)
            
            y_pred = tpot_pipeline.predict(X_test)
            score_dict = {}
            roc_auc = roc_auc_score(y_test, y_pred)
            score_dict['ROC'] = round(roc_auc, 3)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            if tp == 0:
                tp_rate = 0
            else:
                tp_rate = (tp)/(fp + tp)
            if tn == 0:
                tn_rate = 0
            else:    
                tn_rate = (tn)/(fn + tn)
            train_score = tpot_pipeline.score(X_train, y_train)
            test_score = tpot_pipeline.score(X_test, y_test)

            score_dict['Train_score'] = round(train_score,3)
            score_dict['Test_score'] = round(test_score,3)

            score_dict['1'] = round(tp_rate, 3)
            score_dict['0'] = round(tn_rate, 3)
            score_dict['Prediction'] = tpot_pipeline.predict(z)
            score_dict['date'] = z.index
            score_dict['k_val'] = K
            dict_list.append(score_dict)
        
        with open(filepath, 'a') as file:
            for i in range(len(dict_list)):
                scores = dict_list[i]
                mp_tut = {1: 'Long', 0: 'Short'}
                n_predictions = len(scores['Prediction'])
                print(f"{tick}: For K = {scores['k_val']},prediction: {scores['Prediction']}, Confidence {scores['Train_score']}, / {scores['Test_score']}, Prices = {list(round(df_new_p.iloc[-(scores['k_val']):]['Adj Close'], 3))} Shape: {input_shapes}, [0:{scores['0']}, 1:{scores['1']}], AUC_ROC: {scores['ROC']}")

                for j in range(len(scores['Prediction'])):
                    guessDate = df_new_p.index[-(len(scores['Prediction']) - j)]
                    log_data = (tick, #Symbol
                                guessDate.strftime('%Y-%m-%d'),  #Date of Guess
                                df_new_p.iloc[-(n_predictions - j)]['Adj Close'], #Price of Guess
                                scores['k_val'], #N weeks to results
                                mp_tut[scores['Prediction'][j]], #Prediction: "Long/Short"
                                (guessDate + timedelta(weeks = scores['k_val'])).strftime('%Y-%m-%d'), #Date of Result
                                scores['Train_score'], #Train Score
                                scores['Test_score'], #Test Score
                                scores[str(scores['Prediction'][j])], #ConfusionMatrix Score
                                scores['ROC'], #Auc_ROC Score
                                VersionNo #predictor_version
                                )
                    toMySQL(log_data)
                    if scores['Test_score'] >= confi_level and scores[str(scores['Prediction'][j])] >= confu_level and scores['ROC'] >= roc_level: 
                        
                        file.write(f"{tick}, {df_new_p.index[-(n_predictions - j)].month}/{df_new_p.index[-(n_predictions - j)].day}/{df_new_p.index[-(n_predictions - j)].year}, ,{scores['k_val']}, {mp_tut[scores['Prediction'][j]]}, , , , {scores['Train_score']}, {scores['Test_score']}, , {scores[str(scores['Prediction'][j])]}, {scores['ROC']}\n")
        print(f"logged {tick}")
    except IndexError as e:
        print(f"{tick} Fail due to {e}")
        failed.append(tick)

print(list(df_new_p.iloc[-K:]['Adj Close'].index.date))
et = time.time()
print(f"Time elapsed =  {round((et-st)/60,2)} Mins")
print(failed) 