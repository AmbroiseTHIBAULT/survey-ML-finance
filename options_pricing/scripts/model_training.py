from sklearn import gaussian_process
import numpy as np
import os
import sys
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from sklearn.model_selection import train_test_split
chemin_dev = os.path.abspath(os.path.join(os.getcwd(), '../..'))

if chemin_dev not in sys.path:
    sys.path.append(chemin_dev)
from options_pricing.scripts import data_processing as dp 
from options_pricing.scripts import black_scholes as bs

def train_test_gp_model(params, df_options_c):
    dicts_save = {}

    # We instanciate the GaussianProcessRegressor model with the parameters specified in the params dictionary
    gp =[getattr(gaussian_process, type_pro)(**params)for type_pro, params in params['gaussian_process'].items()][0]
    for test_value in params['n_rows']:
        dict_save = {}
        tab_metrics = np.zeros((params['number_time_repeat'], 4))
        tab_metrics_bs = np.zeros((params['number_time_repeat'], 4))
        for i in range(params['number_time_repeat']):
            get_batch_data = getattr(dp, params['type_batch']) 
            X_train, X_test, y_train, y_test, scaler, pca = get_batch_data(df_options_c, 
                                                                                                   params['test_size'], 
                                                                                                   test_value, 
                                                                                                   params['features'], 
                                                                                                   params['standardize'],
                                                                                                    params['pca'],
                                                                                                    params['n_pc']
                                                                                                   )
            if params['pca']: 
                X_test_unpca = pca.inverse_transform(X_test) # for bs model
            else:
                X_test_unpca = X_test
            X_test_unscal = scaler.inverse_transform(X_test_unpca) # for bs model
            gp.fit(X_train, y_train)
            y_pred = gp.predict(X_test)
            mape, mae, mse, r2 = bs.compute_metrics(y_test, y_pred)
            tab_metrics[i] = [mape, mae, mse, r2]

            # for bs evaluation 
            if params['bs_eval']:
                X_test_df, y_test_df = dp.from_np_to_df(X_test_unscal, y_test, params['features'])
                df_price_bs, mape_bs, mae_bs, mse_bs, r2_bs = bs.evaluate_bs_options_pricing(y_test_df, X_test_df)
                
                tab_metrics_bs[i] = [mape_bs, mae_bs, mse_bs, r2_bs]
                mean_metrics_bs = np.mean(tab_metrics_bs, axis=0)
                std_metrics_bs = np.std(tab_metrics_bs, axis=0)
            else: 
                mean_metrics_bs = -1
                std_metrics_bs = -1

        mean_metrics = np.mean(tab_metrics, axis=0)
        std_metrics = np.std(tab_metrics, axis=0)

        dict_save[params['name_run']] = {
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'metrics': tab_metrics,
            'mean_metrics_bs': mean_metrics_bs,
            'std_metrics_bs': std_metrics_bs,
            'metrics_bs': tab_metrics_bs,
            'params': params
            }
        dicts_save[test_value] = dict_save
    return dicts_save




def train_test_dl_model(params, callbacks, df_options_c):
    dicts_save = {}

    for test_value in params['n_rows']:
        dict_save = {}
        tab_metrics = np.zeros((params['number_time_repeat'], 4))
        tab_metrics_bs = np.zeros((params['number_time_repeat'], 4))
        for i in range(params['number_time_repeat']):
            get_batch_data = getattr(dp, params['type_batch']) 
            X_train_val, X_test, y_train_val, y_test, scaler, pca = get_batch_data(df_options_c, 
                                                                                                   params['test_size'], 
                                                                                                   test_value, 
                                                                                                   params['features'], 
                                                                                                   params['standardize'],
                                                                                                    params['pca'],
                                                                                                    params['n_pc']
                                                                                                   )
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1)
            if params['pca']: 
                X_test_unpca = pca.inverse_transform(X_test) # for bs model
            else:
                X_test_unpca = X_test
            X_test_unscal = scaler.inverse_transform(X_test_unpca) # for bs model

            model_with_callbacks = build_model(X_train.shape[1:]) # nombre de features
            history_with_callbacks = model_with_callbacks.fit(
                X_train, 
                y_train, 
                validation_data=(X_val, y_val), 
                epochs=params['epochs'], 
                batch_size=params['batch_size'],
                callbacks=callbacks
            ).history
            y_pred = model_with_callbacks.predict(X_test)
            mape, mae, mse, r2 = bs.compute_metrics(y_test, y_pred)
            tab_metrics[i] = [mape, mae, mse, r2]

            # for bs evaluation 
            if params['bs_eval']:
                X_test_df, y_test_df = dp.from_np_to_df(X_test_unscal, y_test, params['features'])
                df_price_bs, mape_bs, mae_bs, mse_bs, r2_bs = bs.evaluate_bs_options_pricing(y_test_df, X_test_df)
                
                tab_metrics_bs[i] = [mape_bs, mae_bs, mse_bs, r2_bs]
                mean_metrics_bs = np.mean(tab_metrics_bs, axis=0)
                std_metrics_bs = np.std(tab_metrics_bs, axis=0)
            else: 
                mean_metrics_bs = -1
                std_metrics_bs = -1

        mean_metrics = np.mean(tab_metrics, axis=0)
        std_metrics = np.std(tab_metrics, axis=0)

        dict_save[params['name_run']] = {
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'metrics': tab_metrics,
            'mean_metrics_bs': mean_metrics_bs,
            'std_metrics_bs': std_metrics_bs,
            'metrics_bs': tab_metrics_bs,
            'params': params,
            'history': history_with_callbacks
            }
        dicts_save[test_value] = dict_save
    return dicts_save



def build_model(input_shape, learning_rate=0.001):
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')
    x = tfkl.Dense(128, name='dense_1')(input_layer)
    x = tfkl.Activation('leaky_relu', name='relu_1')(x)
    x = tfkl.Dense(128, name='dense_2')(x)
    x = tfkl.Activation('leaky_relu', name='relu_2')(x)
    output = tfkl.Dense(1, name='output')(x)
    output_layer = tfkl.Activation('relu', name='output_activation')(output)
    # we use relu activation function to avoid negative prices

    model = tfk.Model(input_layer, output_layer, name='pricing_model')
    opt = tfk.optimizers.Adam(learning_rate=learning_rate)
    loss = tfk.losses.MeanSquaredError()
    mtr = ['mse']
    model.compile(optimizer=opt, loss=loss, metrics=mtr)
    return model