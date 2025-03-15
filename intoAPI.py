from main import get_prediction
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
                 'BWA','HII','FMC','QRVO','AMTM']

import random
# random.shuffle(sp500_tickers)
count = 0 #sp500_tickers.index('SPGI') + 1
if(__name__ == '__main__'):
    for tick in ['NVDA', 'AAPL'] + sp500_tickers:
        print(count)
        get_prediction(tick, k_min = 2, k_max = 8, logpath= r'Logs\2025-03-10.txt')
        # count += 1
        
        
