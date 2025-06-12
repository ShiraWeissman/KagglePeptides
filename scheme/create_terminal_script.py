import os
import sys
sys.path.append(sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))
from utils.general import *
from utils.schemes import schemes_folder_path, load_schemes_files_names


def main(default_dataset_name=None, model_file_path=None, schemes_files_names=[]):
    parser = argparse.ArgumentParser(description="Create script for running model schemes.")
    parser.add_argument("dataset_name", type=str, default=default_dataset_name,
                        help="Name of dataset.")
    parser.add_argument("model_file_path", type=str, default=model_file_path,
                        help="File path of the model python file.")
    parser.add_argument("schemes_files_names", nargs='*', default=schemes_files_names,
                        help="Files names of YAML schemes files.")
    args = parser.parse_args()
    print("model_file_path:", args.model_file_path)
    if args.dataset_name is None or args.model_file_path is None:
        print("Dataset name or model file path is missing..")
        return
    elif not os.path.isfile(args.model_file_path):
        print("Model file not found..")
        return
    elif not os.path.isdir(os.path.join(schemes_folder_path, args.dataset_name)):
        print("Dataset schemes files folder was not found..")
        return

    if len(args.schemes_files_names) == 0:
        schemes_files_names = load_schemes_files_names(dataset_name=args.dataset_name)
    else:
        schemes_files_names = args.schemes_files_names
    script_path = os.path.join(r'run_logs\terminal_scripts', f"run_model_schemes_{args.dataset_name}.sh")
    if os.path.isfile(script_path):
        os.remove(script_path)
    linux_file = open(script_path, "a")
    linux_file.write('#!/bin/bash\n')
    for scheme_name in schemes_files_names:
        linux_file.write(f'python {args.model_file_path} {args.dataset_name} {scheme_name};\n')
    linux_file.close()
    print(f"The script: run_model_schemes_{args.dataset_name}.sh, in run_logs\\terminal_scripts folder")


if __name__ == '__main__':
    main(default_dataset_name=None, model_file_path=None, schemes_files_names=[])