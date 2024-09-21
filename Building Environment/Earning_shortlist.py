luist = []
import yfinance as yf
import pandas as pd
import numpy as np
import time as time
from datetime import datetime
stock_ticks = ['ABBV', 'TJX', 'AMD', 'GE','RSG','KO', 'V', 'PG', 'CL', 'L', 'MMC', 'ROP', 'MCD', 'TMUS', 'WM', 'MA', 'COR', 
                 'YUM', 'PM', 'TJX', 'JNJ', 'KMB', 'HON', 'ITW', 'ATO', 'PEP', 'FI', 'ICE', 'AME', 
                 'LIN', 'KMI', 'WMT', 'JPM', 'LMT', 'MDLZ', 'GD', 'OTIS', 'CHD', 'DUK', 'HIG', 'MO',
                 'CSX', 'AJG', 'WMB', 'STZ', 'CME', 'APH', 'MRK', 'SYY', 'MSI', 'ED', 'AMP', 'ABT', 
                 'VRSK', 'AIG', 'COST', 'VRSN', 'ECL', 'BRO', 'SO', 'PEG', 'LH', 'PPL', 'ABBV', 'CB', 
                 'HOLX', 'BR', 'AVY', 'CTAS', 'IEX', 'MCK', 'KDP', 'TRV', 'DRI', 'CNP', 'ADP', 'FE',
                 'CSCO', 'KHC', 'EA', 'PAYX', 'SRE', 'MDT', 'CMS', 'PRU', 'GIS', 'OKE', 'CBOE', 'LNT',
                 'HD', 'ETR', 'DGX', 'AFL', 'WEC', 'BK', 'JKHY', 'LYB', 'SPGI', 'HLT', 'XYL', 'DOV', 
                 'EXPD', 'VICI', 'PPG', 'ORLY', 'SHW', 'BSX', 'NI', 'LDOS', 'DTE', 'MCO', 'BDX', 'O',
                 'BLK', 'REGN', 'WRB', 'DOW', 'UNP', 'PFG', 'FDS', 'PTC', 'CPRT', 'EVRG', 'TEL', 'CVX', 
                 'HSY', 'AON', 'AVB', 'AAPL', 'FAST', 'INVH', 'MET', 'KR', 'PNW', 'WAB', 'AEP', 'REG', 
                 'TDG', 'VMC', 'K', 'CAG', 'NDSN', 'CTSH', 'IBM', 'SYK', 'AEE', 'PCG', 'FTV', 'XOM',
                 'FFIV', 'EIX', 'EQR', 'GWW', 'WELL', 'ROST', 'MSFT', 'EXC', 'PKG', 'AZO', 'J', 'TMO', 
                 'PCAR', 'NOC', 'ZBH', 'ACN', 'CINF', 'MNST', 'MAR', 'STE', 'LOW', 'AXP', 'CAH', 'HSIC',
                 'RJF', 'LHX', 'BMY', 'GS', 'FRT', 'CDW', 'ELV', 'WTW', 'NWSA', 'C', 'AOS', 'NWS', 'SJM',
                 'TRGP', 'WY', 'GLW', 'IR', 'SNA', 'MLM', 'EMR', 'CMCSA', 'GILD', 'FOX', 'TAP', 'AWK', 
                 'AMCR', 'TSN', 'SPG', 'WFC', 'MAA', 'XEL', 'RTX', 'COP', 'UNH', 'OMC', 'ACGL', 'DHR', 
                 'FOXA', 'CMG', 'BG', 'CMI', 'BKR', 'TSCO', 'TDY', 'HCA', 'AIZ', 'OXY', 'ALL', 'FANG', 
                 'CPB', 'UDR', 'AKAM', 'IRM', 'ESS', 'NDAQ', 'PSA', 'HII', 'ROL', 'TTWO', 'RHI', 'PNR',
                 'CTRA', 'EG', 'ALLE', 'NSC', 'DE', 'VZ', 'EMN', 'BAC', 'VRTX', 'UPS', 'UHS', 'CLX', 
                 'STT', 'TXN', 'T', 'PH', 'EOG', 'CPT', 'COO', 'ETN', 'NVR', 'BIIB', 'HST', 'TYL', 'MS',
                 'TXT', 'MKC', 'FIS', 'PFE', 'GRMN', 'PSX', 'CPAY', 'INCY', 'AMGN', 'TT', 'FDX', 'JCI',
                 'KVUE', 'GPC', 'LKQ', 'COF', 'MAS', 'MOH', 'D', 'CNC', 'AMT', 'EQIX', 'IT', 'GE', 'HES',
                 'CDNS', 'PNC', 'KIM', 'TROW', 'PLD', 'CCI', 'JBHT', 'INTU', 'PGR', 'HUBB', 'VTR', 'SBUX',
                 'DPZ', 'BKNG', 'DVN', 'VTRS', 'ES', 'IPG', 'HWM', 'SLB', 'DD', 'DLR', 'EBAY', 'VLTO', 
                 'ZTS', 'ADSK', 'IQV', 'DIS', 'NKE', 'NRG', 'MPC', 'KEYS', 'ISRG', 'CBRE', 'ADI', 'SNPS',
                 'NUE', 'HAL', 'CE', 'A', 'GPN', 'SYF', 'WRK', 'BAX', 'VLO', 'CAT', 'HBAN', 'APD', 'LYV', 
                 'GOOG', 'TFX', 'AMZN', 'CI', 'HRL', 'GOOGL', 'BEN', 'NTRS', 'MSCI', 'WYNN', 'NEE', 'EW', 
                 'PWR', 'BALL', 'CARR', 'POOL', 'STLD', 'CF', 'DOC', 'HPQ', 'IP', 'CHRW', 'MMM', 'GM', 'DAL', 
                 'LVS', 'MRO', 'IDXX', 'SBAC', 'BBY', 'CSGP', 'JNPR', 'ULTA', 'SCHW', 'TRMB', 'MTB', 'NTAP',
                 'FICO', 'BIO', 'WAT', 'EFX', 'LLY', 'ANSS', 'RL', 'WST', 'LEN', 'NOW', 'ROK', 'GEHC', 'HPE',
                 'QCOM', 'NXPI', 'DLTR', 'PHM', 'CTVA', 'SWK', 'TFC', 'BX', 'BA', 'DHI', 'RF', 'FITB', 'CVS',
                 'USB', 'SWKS', 'RVTY', 'EXR', 'CRL', 'MGM', 'ODFL', 'MTD', 'STX', 'RCL', 'ADBE', 'EQT', 'IVZ',
                 'APA', 'LW', 'BWA', 'KLAC', 'MCHP', 'ORCL', 'AXON', 'TGT', 'GEN', 'HUM', 'MHK', 'QRVO', 'TPR',
                 'AMAT', 'F', 'ADM', 'LRCX', 'MOS', 'HAS', 'IFF', 'ARE', 'VST', 'CRM', 'LULU', 'WDC', 'ABNB', 
                 'FCX', 'TER', 'DAY', 'APTV', 'UBER', 'AVGO', 'CEG', 'CFG', 'NEM', 'CHTR', 'DG', 'TECH', 'NFLX',
                 'DFS', 'DVA', 'PYPL', 'LUV', 'META', 'AES', 'URI', 'MKTX', 'MU', 'MTCH', 'SOLV', 'BBWI', 'KMX',
                 'DECK', 'WBA', 'CMA', 'AAL', 'UAL', 'KEY', 'INTC', 'RMD', 'CTLT', 'ILMN', 'ZBRA', 'DXCM', 'BXP',
                 'FTNT', 'JBL', 'MPWR', 'CZR', 'ANET', 'EL', 'NVDA', 'PODD', 'EXPE', 'ON', 'BLDR', 'FMC', 'ALGN',
                 'CCL', 'AMD', 'ETSY', 'EPAM', 'FSLR', 'PANW', 'GNRC', 'WBD', 'TSLA', 'NCLH', 'ALB', 'PAYC', 'GEV',
                 'MRNA', 'PARA', 'ENPH', 'GL', 'SMCI', 'BRK.B', 'BF.B']


def ccal(df):
    try:
        yoyIncrease = df.dropna(subset = ['EPS Estimate'])['EPS Estimate'][0] > df.dropna(subset = ['Reported EPS'])['Reported EPS'][3]
        supriseConsi = ((df['Surprise(%)'].dropna() > 0).mean()) == 1 
        increaseEarnings = (df['EPS Estimate'].dropna()[0] > df['Reported EPS'].dropna()[0])
        conditions = yoyIncrease and supriseConsi and increaseEarnings       
        
        peakEarnings = df['Reported EPS'].dropna()[1:4].max() <= df['EPS Estimate'].dropna()[0]
    except:
        print(f'{ticker}, Failed')
        return 0
    if conditions:
        if peakEarnings:
            return 1, 1
        else:
            return 1, 0
    else:
        return 0, 0



for ticker in stock_ticks:
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.earnings_dates
        rating = ccal(data)
        if rating == (1, 1):
            luist.append([ticker, data[data['Reported EPS'].isna()].index[-1].date(), True])
        elif rating == (1, 0):
            luist.append([ticker, data[data['Reported EPS'].isna()].index[-1].date(), False])
    except KeyError:
        print(f'{ticker}, Failed')


output = pd.DataFrame(luist, columns = ['tickers', 'next_earnings', 'Peak']).sort_values(['Peak', 'next_earnings'], ascending = [False, True])
print(output)
output_path = r'C:\Users\Jerome\Desktop\Jerome_Ground\Stocker_Git\AI_Earning_Scraper\08-17-24.csv'
output.to_csv(output_path, index = False)