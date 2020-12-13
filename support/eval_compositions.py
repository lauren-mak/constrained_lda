import click
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename, join
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import matthews_corrcoef
from seaborn import boxplot, stripplot

@click.group()
def main():
    pass

def calc_mcc(actual_alleles, predicted_alleles, threshold):
    num_actual_strains = actual_alleles.shape[0]
    num_predicted_strains = predicted_alleles.shape[0]
    num_positions = actual_alleles.shape[1]

    mcc_mtx = np.zeros((num_actual_strains, num_predicted_strains))
    closest_mcc = []
    closest_to_actual = []
    num_reconstructed_strains = 0.0
    for i in range(num_actual_strains):
        for j in range(num_predicted_strains): 
            mcc_mtx[i][j] = matthews_corrcoef(actual_alleles[i], predicted_alleles[j])
        largest_mcc = np.amax(mcc_mtx[i])
        largest_mcc_idx = np.where(mcc_mtx[i] == largest_mcc)
        closest_mcc.append(largest_mcc)
        closest_to_actual.append(largest_mcc_idx)
        if 1 - np.linalg.norm(actual_alleles[i] - predicted_alleles[largest_mcc_idx]) / num_positions  > threshold:
            num_reconstructed_strains += 1.0
    return np.mean(closest_mcc), closest_to_actual, num_reconstructed_strains / num_actual_strains

def calc_jsd(actual_freqs, predicted_freqs, closest_to_actual):
    jsd = []
    num_pools = actual_freqs.shape[0]
    num_actual_strains = actual_freqs.shape[1]
    for i in range(num_pools):
        matched_strain_freqs = np.zeros(num_actual_strains)
        for j in range(num_actual_strains): 
            matched_strain_freqs[j] = predicted_freqs[i][closest_to_actual[j]]
        jsd.append(jensenshannon(predicted_freqs[i], matched_strain_freqs))
    return np.mean(jsd)

def make_summary_stats(actual_prefix, predict_prefix, num_tests, threshold):
    mcc_lst = []
    crr_lst = []
    jsd_lst = []
    for t in range(num_tests):
        test_actual_prefix = '.'.join(actual_prefix, str(t))
        test_predict_prefix = '.'.join(predict_prefix, str(t))
        mcc, closest_to_actual, crr = calc_mcc(test_actual_prefix + '.topic_wrd.csv', test_predict_prefix + '.topic_wrd.csv', threshold)
        mcc_lst.append(mcc)
        crr_lst.append(crr)
        jsd_lst.append(calc_jsd(test_actual_prefix + '.doc_topic.csv', test_predict_prefix + '.doc_topic.csv', closest_to_actual))
    return mcc_lst, crr_lst, jsd_lst

def make_graph(arr, names, stat, results_prefix): 

    df_list = []
    for i, a in enumerate(arr):
        df_list.append(pd.DataFrame([names[i] * len(a), a], index = names).transpose())
    df = pd.concat(df_list)
    df.columns = ['Parameters (k, eta)', stat]

    plt.figure(figsize=(16,10), dpi= 120)
    boxplot(x = 'Parameters (k, eta)', y = stat, data = df, hue = 'Parameters')
    stripplot(x = 'Parameters (k, eta)', y = stat, data = df, color='black', size = 3, jitter = 1)
    plt.savefig(results_prefix + '.' + stat + '.png')

# File-name convention (actual): actual_dir/numvar_numstr.attempt.mtx-type.csv
# File-name convention (predicted): predict_dir/numvar_numstr.train-test.k_eta.attempt.mtx-type.csv
@main.command('eval_compositions')
@click.argument('actual_dir') 
@click.argument('predict_dir') 
@click.argument('results_dir') 
@click.argument('prefix') # numvar_numstr
@click.argument('data_type') # train or test
@click.argument('k_values')
@click.argument('eta_values')
@click.argument('num_tests') # TODO set default at 50
@click.argument('threshold')
def eval_compositions(prefix, data_type, k_values, eta_values, num_tests):
    compare_mcc = []
    compare_crr = []
    compare_jsd = []
    k_list = k_values.split(',')
    eta_list = eta_values.split(',')
    param_names = []
    for k in k_list:
        for e in eta_list:
            param_names.append('k = ' + str(k) + ', eta = ' + str(e))
            actual_prefix = actual_dir + '/' + prefix
            predict_prefix = predict_dir + '/' + '.'.join(prefix, data_type, str(k) + '_' + str(e))
            mcc_lst, crr_lst, jsd_lst = make_summary_stats(actual_prefix, predict_prefix, int(num_tests), float(threshold)) 
            compare_mcc.append(mcc_lst)
            compare_crr.append(crr_lst)
            compare_jsd.append(jsd_lst)
    results_prefix = results_dir + '/' + prefix
    make_graph(closest_mcc, param_names, 'Matthews_Corr_Coef', results_prefix) 
    make_graph(closest_jsd, param_names, 'Jensen_Shannon_Div', results_prefix) 
    make_graph(crr, param_names, 'Correct_Recon_Rate', results_prefix) 

if __name__ == '__main__':
    main()
