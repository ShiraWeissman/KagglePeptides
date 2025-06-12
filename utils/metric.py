from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.schemes import *


def calculate_regression_metric(data):
    predictions, labels = data
    mse = mean_squared_error(predictions, labels)
    mae = mean_absolute_error(predictions, labels)
    r2 = r2_score(predictions, labels)
    return {"mse": mse, "mae": mae,  "r2": r2}


def generate_schemes_metric_csv(run_name,  params=None, schemes=None):
    if schemes == None:
        schemes = [s for s in os.listdir(os.path.join(folder_path,'run_logs', 'schemes')) if s.split('_')[0]==run_name]
    scheme_metric_list = []
    for s in schemes:
        scheme = load_scheme(os.path.join(folder_path,'run_logs', 'schemes', s))
        preds = load_predictions(scheme)
        train_metric = calculate_regression_metric_metric(preds['y_train'], preds['y_train_pred'], scheme)
        train_metric = dict((k + '_train', train_metric[k]) for k in train_metric)
        scheme.update(train_metric)
        test_metric = calculate_metric(preds['y_test'], preds['y_test_pred'], scheme)
        test_metric = dict((k + '_test', test_metric[k]) for k in test_metric)
        scheme.update(test_metric)
        if 'y_valid' in preds:
            valid_metric = calculate_metric(preds['y_valid'], preds['y_valid_pred'], scheme)
            valid_metric = dict((k + '_valid', valid_metric[k]) for k in valid_metric)
            scheme.update(valid_metric)
        scheme_metric_list.append(scheme)
    df = pd.DataFrame.from_records(scheme_metric_list)
    if bool(params):
        df = df[params + list(train_metric.keys()) + list(test_metric.keys())]
    with open(os.path.join(folder_path, 'results', 'metric_summary', f'{run_name}.csv'), 'w') as f:
        df.to_csv(f, index=False)

def choose_best_results(results_df):
    cols = results_df.columns
    model_type = results_df['model_type'].unique()[0]
    if model_type == 'regressor':
        results_df = results_df[results_df['mse_train'] < 1000]
        results_df = results_df[results_df['mse_test'] < 1000]
        results_df = results_df[results_df['R2_score_train'] > 0]
        results_df = results_df[results_df['R2_score_test'] > 0]
        if not results_df.empty:
            best_result = results_df[results_df['R2_score_test']==results_df['R2_score_test'].max()]
            print(best_result)
            best_scheme = best_result.to_dict(orient='records')[0]
            best_scheme_path = os.path.join(folder_path, 'results', 'best_schemes', f'{best_scheme["run_name"]}.json')
            verify_folder(best_scheme_path)
            save_scheme(best_scheme, best_scheme_path)
        else:
            print('no good results')

    print()

if __name__=='__main__':
    good_schemes = ['CNNRegressor_pytorch_' + f for f in
                    os.listdir(r'C:\Users\Shira\Desktop\postdoc\MLFramework\results\predictions') if f[:5] in ['oeiyv', 'uzbau']]
    run_name = 'oeiyv'
    generate_schemes_metric_csv(run_name=run_name, params=None, schemes=good_schemes)
    with open(os.path.join(folder_path, 'results', 'metric_summary', f'{run_name}.csv'), 'r') as f:
        df = pd.read_csv(f)
    #choose_best_results(df)