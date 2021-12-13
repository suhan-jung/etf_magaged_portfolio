import pandas as pd

def drawdown(return_series: pd.Series):
    """
    takes a time series of asset returns.
    returns a DataFrame with columns for the wealth index,
    the previous peaks, and the percentage drawdown
    
    Args:
        return_series (pd.Series): [description]
    """
    wealth_index = 1000 * (return_series + 1).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdown
    })

def portfolio_return(weights, returns):
    """
    weights -> returns
    """
    return weights.T @ returns

def portfolio_volatility(weights, cov_matrix):
    """
    weights -> volatility
    """
    return (weights.T @ cov_matrix @ weights) ** 0.5