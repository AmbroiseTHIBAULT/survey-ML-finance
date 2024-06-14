import numpy as np 
from scipy.stats import norm

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

def bs_options_pricing(S, K, r, T, sigma, option_type='Call'):
    # We apply the Black-Scholes formula to calculate the price of the option
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # In the case the option price is NaN, we want to know the parameters that led to this
    if np.isnan(option_price):
        print("S: ", S)
        print("K: ", K)
        print("r: ", r)
        print("T: ", T)
        print("sigma: ", sigma)

    return option_price



def compute_metrics(y_true, y_pred):
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    except Exception as e:
        print("An error occurred:", e)
        print("y_true:", y_true)
        print("y_pred:", y_pred)
        return None

    return mape, mae, mse, r2


def evaluate_bs_options_pricing(df_price, df_options):
    """
    Evaluate Black-Scholes options pricing model
    df_price: DataFrame with stock price data
    df_options: DataFrame with options data
    """

    # Calculate the price of each option
    df_price.reset_index(drop=True, inplace=True)
    df_price['price_bs'] = df_options.apply(lambda row: bs_options_pricing(row['stock_price'], row['strike'], row['interest_rate'], row['time_to_maturity'], row['vol']), axis=1)

    y_true = df_price['price'].values
    # check if NaN values are present in the price_bs column

    y_pred = df_price['price_bs'].values
    # Calculate metrics 
    print('Black-Scholes Options Pricing Model')
    
    mape, mae, mse, r2 = compute_metrics(y_true, y_pred)


    return df_price, mape, mae, mse, r2



