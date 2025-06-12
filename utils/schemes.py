from utils.general import *
import shutil
import yaml
from itertools import product


schemes_folder_path = os.path.join(root_path, r'run_logs\schemes')
best_schemes_path = os.path.join(root_path, r'results\best_schemes')



def expand_scheme(scheme):
    """
    Expands a nested dictionary containing lists into a list of dictionaries,
    generating all possible combinations where list values are replaced by single elements.

    Args:
        scheme (dict): A nested dictionary potentially containing lists as values.

    Returns:
        list: A list of nested dictionaries representing all possible combinations.
    """

    items = []
    for key, value in scheme.items():
        if key == 'layers':
            expended_layers = []
            for layer in value:
                expended_layers.append(expand_scheme(layer))
            items.append([(key, expanded) for expanded in list(product(*expended_layers))])
        elif isinstance(value, list):
            items.append([(key, item) for item in value])
        elif isinstance(value, dict):
            items.append([(key, expanded) for expanded in expand_scheme(value)])
        else:
            items.append([(key, value)])

    expanded_schemes = []
    for i, combination in enumerate(product(*items)):
        new_scheme = {}
        for key_value_pairs in combination:
            if isinstance(key_value_pairs[1], dict):
                new_scheme[key_value_pairs[0]] = key_value_pairs[1]
            else:
                new_scheme[key_value_pairs[0]] = key_value_pairs[1]
        expanded_schemes.append(new_scheme)

    return expanded_schemes

def expended_schemes_to_folder(scheme):
    expanded_schemes = expand_scheme(scheme)
    for i in range(len(expanded_schemes)):
        expanded_schemes[i]['data']['run_name'] += f'_{i}'
        expanded_schemes[i]['model']['run_name'] += f'_{i}'
        save_scheme(expanded_schemes[i],
                     os.path.join(schemes_folder_path,
                                  expanded_schemes[i]['data']['dataset'],
                                  f"{expanded_schemes[i]['model']['model_name']}_{expanded_schemes[i]['model']['run_name']}.yaml"))


# def generate_hyperparameter_grid_schemes_list(scheme):
#     print("Collecting schemes..")
#     keys = list(scheme.keys())
#     for k in keys:
#         scheme[k] = [scheme[k]]
#     df = pd.DataFrame.from_dict(scheme)
#     columns_with_list = [col for col in df.columns if isinstance(df.loc[0, col], list)]
#     for col in columns_with_list:
#         df = df.explode(column=col)
#     print('Number of schemes:', len(df.index))
#     print('Transferring schemes to json format..')
#     scheme_list = df.to_dict('records')
#     for i in range(len(scheme_list)):
#         scheme_list[i]["run_name"] = scheme_list[i]["run_name"] + f'_{i}'
#     print(f"Saving schemes to run_logs\schemes\{scheme['model_name']} folder..")
#     for s in scheme_list:
#         save_scheme(s, os.path.join(schemes_folder_path, s['model_name'], f'{s["dataset"]}_{s["run_name"]}_model.json'))
#     return scheme_list


def load_schemes_files_names(dataset_name, included_text=[], folder_path=schemes_folder_path):
    print("Loading schemes file names..")
    schemes = [n for n in os.listdir(os.path.join(folder_path, dataset_name)) if sum([t in n for t in included_text]) == len(included_text)]
    return schemes


def save_scheme(scheme, scheme_file_path):
    verify_folder(scheme_file_path)
    with open(scheme_file_path, 'w') as f:
        yaml.safe_dump(scheme, f)


def load_scheme(scheme_file_path):
    def construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.add_constructor('tag:yaml.org,2002:python/tuple', construct_python_tuple)
    try:
        with open(scheme_file_path, 'r') as f:
            return yaml.full_load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {scheme_file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None


def copy_schemes_to_another_folder(dataset_name, origin_folder, destination_folder, included_text=[]):
    schemes_file_names = load_schemes_files_names(dataset_name, included_text=included_text, folder_path=origin_folder)
    for i, n in enumerate(schemes_file_names):
        print(f"Copying scheme { i+ 1} out of {len(schemes_file_names)}..")
        shutil.copy(os.path.join(origin_folder, dataset_name, n), os.path.join(destination_folder,dataset_name, n))


def move_schemes_to_another_folder(dataset_name, origin_folder, destination_folder, included_text=[]):
    schemes_file_names = load_schemes_files_names(dataset_name, included_text=included_text, folder_path=origin_folder)
    for i, n in enumerate(schemes_file_names):
        print(f"Moving scheme {i + 1} out of {len(schemes_file_names)}..")
        shutil.move(os.path.join(origin_folder, dataset_name,  n), os.path.join(destination_folder, dataset_name, n))


def delete_schemes_from_folder(schemes_folder, dataset_name, included_text=[]):
    schemes_file_names = load_schemes_files_names(dataset_name, included_text=included_text, folder_path=schemes_folder)
    for i, n in enumerate(schemes_file_names):
        print(f"Deleting scheme {i + 1} out of {len(schemes_file_names)}..")
        os.remove(os.path.join(schemes_folder,dataset_name, n))


if __name__ == '__main__':
    pass
