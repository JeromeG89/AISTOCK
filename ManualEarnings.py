import pandas as pd
import os
from dotenv import load_dotenv
from alpha_vantage.fundamentaldata import FundamentalData


load_dotenv()
AVkey = os.getenv("alpha_vantageAPI")
fd = FundamentalData(AVkey)
def getEarnings(ticker):

    earnings_data, _ = fd.get_earnings_quarterly(ticker)
    
    earnings_data['reportedDate'] = pd.to_datetime(earnings_data['reportedDate'])
    earnings_data.index = earnings_data['reportedDate'].dt.strftime('%Y-%m-%d')
    earnings_data = earnings_data.rename(columns = {'reportedEPS': 'Reported EPS'})
    earnings_data['Reported EPS'] = earnings_data['Reported EPS'].dropna().astype("float")
    return earnings_data['Reported EPS']

