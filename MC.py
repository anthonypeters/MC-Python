import pandas as pd
import numpy as np
import statistics
import datetime as dt
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

#### Read input from csv
columns = ['Tickers', 'Weights']
df = pd.read_csv(filepath_or_buffer = 'ticksandweights.csv',  names=columns)
tickers = df.Tickers.to_list()
weights = df.Weights.to_list()

#### Function to grab data from Quandl
def get_data (stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov() 
    return meanReturns, covMatrix 
####

#### Set dates and create dataframes with tickers
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)
meanReturns, covMatrix = get_data(tickers, startDate, endDate)
####


################ Monte Carlo Method ################
mc_sims = 1000
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 1000000

L = np.linalg.cholesky(covMatrix)
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel("Portfolio Value in ($)")
plt.xlabel("Number of Days")
plt.title(str(mc_sims) + " Simulations")
plt.show()

def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distirbution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")

portfolioResults = pd.Series(portfolio_sims[-1,:])

VaR = initialPortfolio - mcVaR(portfolioResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portfolioResults, alpha=5)
print('VaR ${}'.format(round(VaR, 2)))
print('CVaR ${}'.format(round(CVaR, 2)))

#### Calculating/Locating important metrics
# avg_return = 0
# max_return = 0
# min_return = 0
# median_return = 0

# returns_list = []
# for i in portfolio_sims:
#     returns_list.append((i[len(i)-1] - i[0]) / i[0])

# cumulative = 0
# for j in returns_list:
#     cumulative += j

# avg_return = cumulative / mc_sims
# max_return = max(returns_list)
# min_return = min(returns_list)
# median_return = statistics.median(returns_list)

# print("Avg return: " + str(avg_return))
# print("Min return: " + str(min_return))
# print("Median return: " + str(median_return))
# print("Max return: " + str(max_return))












