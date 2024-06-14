import matplotlib.pyplot as plt

def plot_model_comparison_from_dict(dicts_test, params, method, type_subset):

    metrics = ['MAPE', 'MAE', 'MSE', 'R2']
    
    test_values = list(dicts_test.keys())
    gp_results = [dicts_test[key][params['name_run']]['mean_metrics'] for key in test_values]
    bs_results = [dicts_test[key][params['name_run']]['mean_metrics_bs'] for key in test_values]

    num_metrics = len(metrics)

    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 15), sharex=True)
    
    for i in range(num_metrics):
        gp_metric_values = [res[i] for res in gp_results]
        bs_metric_values = [res[i] for res in bs_results]
        
        axs[i].plot(test_values, gp_metric_values, label=method, marker='o')
        axs[i].plot(test_values, bs_metric_values, label='Black-Scholes', marker='x')
        axs[i].set_ylabel(metrics[i])
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Dataset size')
    plt.suptitle(f'Model Comparison: {method} vs Black-Scholes for {type_subset} type of training data', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_loss_mse(history):
    ignore = 0

    plt.figure(figsize=(21, 4))


    plt.plot(history['loss'][ignore:], label='Training loss', alpha=.2, color='red')
    plt.plot(history['val_loss'][ignore:], label='Validation loss', alpha=.8, color='blue')
    plt.title('Loss')
    plt.yscale('log')
    plt.legend()

    plt.grid(alpha=.3)


    plt.figure(figsize=(21, 4))


    plt.plot(history['mse'][ignore:], label='Training MSE', alpha=.2, color='red')
    plt.plot(history['val_mse'][ignore:], label='Validation MSE', alpha=.8, color='blue')
    plt.title('Mean Squared Error')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=.3)


    plt.show()

