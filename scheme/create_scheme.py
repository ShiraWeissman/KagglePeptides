import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from utils.general import *
from utils.schemes import load_scheme, expended_schemes_to_folder




def main(default_dataset_name=None, default_scheme_path=None):
    parser = argparse.ArgumentParser(description="Load scheme from YAML file.")
    parser.add_argument("dataset_name", type=str, default=default_dataset_name,
                        help="Name of dataset.")
    parser.add_argument("scheme_file_path", type=str, default=default_scheme_path,
                        help="File path of YAML scheme file.")
    args = parser.parse_args()
    try:
        scheme = load_scheme(os.path.join(root_path, args.scheme_file_path))
    except FileNotFoundError:
        print(f"Error: File '{args.scheme_file_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in '{args.scheme_file_path}': {e}")
        return
    expended_schemes_to_folder(scheme)


if __name__ == '__main__':
    main(default_dataset_name=None, default_scheme_path=None)
