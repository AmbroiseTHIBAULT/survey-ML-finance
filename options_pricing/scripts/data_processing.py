import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
import os
import sys
chemin_dev = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if chemin_dev not in sys.path:
    sys.path.append(chemin_dev)
from options_pricing.scripts import black_scholes as bs

# We need this dictionnary to associate the interest rate to the time to maturity of the option
DICT_CORRESPONDANCE_JOUR_IR = {
    '1 Mo': 1/12,
    '2 Mo': 2/12,
    '3 Mo': 3/12,
    '6 Mo': 6/12,
    '1 Yr': 1,
    '2 Yr': 2,
    '3 Yr': 3,
    '5 Yr': 5,
    '7 Yr': 7,
    '10 Yr': 10,
}


def create_batch_data_idx_random(df, batch_size, features):
    """
    Create a batch of data with random index
    df: DataFrame with options data
    batch_size: Batch size
    """
    random_index = random.randint(0, len(df) - batch_size)
    df_options_batch = df.iloc[random_index:random_index + batch_size]
    return df_options_batch[features], df_options_batch['price']


def create_batch_data_random(df, batch_size, features):
    """
    Here we do not take temporality into account
    """
    df_options_batch = df.sample(n=batch_size)
    return df_options_batch[features], df_options_batch['price']


def create_batch_data_by_act_symbol(df, batch_size, features):
    """
    Here we take the data related to n act_symbols where the number of options of the n act_symbols is the closest to batch_size
    """
    act_symbol_counts = df['act_symbol'].value_counts()
    sorted_act_symbols = act_symbol_counts.index.tolist()
    cumulative_act_symbol_counts = act_symbol_counts.cumsum()
    index = cumulative_act_symbol_counts.searchsorted(batch_size)
    number_act = index + 1 
    act_symbols = sorted_act_symbols[:number_act]
    df_options_batch = df[df['act_symbol'].isin(act_symbols)]
    return df_options_batch[features], df_options_batch['price']


def create_train_test_set(X, y, test_size):
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def create_train_test_set_sep(df, test_value, train_size, features):
    # We generate two differents random train test set 
    X_test, y_test = create_batch_data_idx_random(df, test_value, features)
    X_train, y_train = create_batch_data_idx_random(df, train_size, features)
    return X_train, y_train, X_test, y_test


def create_train_test_set_idx(df, test_size, batch_size, features):
    X, y = create_batch_data_idx_random(df, batch_size, features)
    return create_train_test_set(X, y, test_size)


def create_train_test_set_random(df, test_size, batch_size, features):
    X, y = create_batch_data_random(df, batch_size, features)
    return create_train_test_set(X, y, test_size)


def create_train_test_set_by_act_symbol(df, test_size, batch_size, features):
    X, y = create_batch_data_by_act_symbol(df, batch_size, features)
    return create_train_test_set(X, y, test_size)


def from_np_to_df(X, y, features):
    df_X = pd.DataFrame(X, columns=features)
    df_y = pd.DataFrame(y, columns=['price'])
    return df_X, df_y



def compute_pca(tab_x, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(tab_x)
    return principal_components

def find_risk_free_rate(df_r, date, maturity):
    """
    This function is not used for now and is incomplete
    """
    # Filter data
    closest_key = min(DICT_CORRESPONDANCE_JOUR_IR, key=lambda x: abs(DICT_CORRESPONDANCE_JOUR_IR[x] - maturity))

    # Calculate average risk-free rate
    r = df_r['rate'].mean()
    return r


def process_data(df_options, df_r, df_stock):
    """
    Process data for Black-Scholes options pricing model
    df_options: DataFrame with options data
    df_r: DataFrame with risk-free rate data
    df_stock: DataFrame with stock data
    """

    # The price to predict is the average of bid and ask
    df_options['price'] = (df_options['bid'] + df_options['ask']) / 2

    df_options = df_options.merge(df_stock[['date', 'act_symbol', 'open']], on=['date', 'act_symbol'], how='left')
    df_options.rename(columns={'open': 'stock_price'}, inplace=True)

    df_options.dropna(subset=['stock_price'], inplace=True) # Drop rows with missing stock price
    df_options = df_options[df_options['price'] != 0] # Drop rows with price equal to 0
    # Creating the time to maturity column that is the number of days between the sell date and the expiration divided by 365
    df_options['time_to_maturity'] = (pd.to_datetime(df_options['expiration']) -  pd.to_datetime(df_options['date'])).dt.days / 365
    
    # Converting the format to have the same format as the options dataframe
    df_r['Date'] = pd.to_datetime(df_r['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    df_r.rename(columns={'Date': 'date'}, inplace=True)

    df_options['interest_rate'] = None # Initialize interest rate column

    # This row associate the corresponding interest rate to an option row
    # we take as the interest rate maturity the closest to the time to maturity of the option 
    # We can then associate the interest rate of the specific date that have the specific maturity
    df_options['interest_rate'] = df_options.apply(lambda row: df_r.loc[df_r['Date'] == row['date'], min(DICT_CORRESPONDANCE_JOUR_IR, key=lambda x: abs(DICT_CORRESPONDANCE_JOUR_IR[x] - row['temps_to_maturity']))].iloc[0], axis=1)

    # Saving the processed data 
    df_options.to_csv('../../data/options/options_processed.csv', index=False)
    return df_options



def generate_data_bs(df_options):
    # From the processed data we generate the price of the options using the Black-Scholes model
    df_options.drop('price', axis=1, inplace=True) # We drop the real price of the options

    # For each row we calculate the price of the option using the Black-Scholes model
    df_options['price'] = df_options.apply(lambda row: bs.bs_options_pricing(row['stock_price'], row['strike'], row['interest_rate'], row['time_to_maturity'], row['vol']), axis=1)

    # We keep only the columns that are useful related to the Black-Scholes model
    df_options = df_options[['stock_price', 'strike', 'interest_rate', 'time_to_maturity', 'vol', 'price']]
    return df_options
