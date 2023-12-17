import sys
import os
sys.path.insert(0, os.getcwd() + '/')

import argparse
import difflib
import pickle
import json
import numpy as np
from tqdm import tqdm
from src.augmentation.generate_refactoring import generate_adversarial
from src.augmentation.refactoring_methods import *
from multiprocessing import Pool


def download_dataset(language, data_path):
    os.makedirs(data_path, exist_ok=True)
    url = f'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip'
    os.system(f'wget {url} -O {data_path}/{language}.zip')
    os.system(f'unzip {data_path}/{language}.zip -d {data_path}')
    os.system(f'rm -r {data_path}/{language}')
    os.system(f'rm {data_path}/{language}_licenses.pkl')
    os.system(f'rm {data_path}/{language}.zip')


def export_functions(language, data_path):
    os.makedirs(f'{data_path}/{language}', exist_ok=True)

    with open(f'{data_path}/{language}_dedupe_definitions_v2.pkl', 'rb') as f:
        pkl_file = pickle.load(f)

        counter = 0
        for dct in tqdm(pkl_file):
            tokens = 'public class DummyClass {\n\t'
            tokens += dct['function']
            tokens += '\n}'

            with open(f'{data_path}/{language}/{counter}.java', 'w') as fw:
                fw.write(tokens)
            
            counter += 1


def augment_functions(language, data_path, validation_frac):
    global source_path
    global val_frac
    global training_set
    global valid_set

    val_frac = validation_frac
    source_path = f'{data_path}/{language}'

    output_path = f'{data_path}/{language}_transformed'
    
    if not os.path.exists(source_path):
        print(f'Please create a source folder {data_path}/{language} and place your original java files there')
        sys.exit()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    training_set = open(f'{output_path}/train.jsonl', 'wt')
    valid_set = open(f'{output_path}/valid.jsonl', 'wt')

    f_names = [int(x.split('.')[0]) for x in os.listdir(source_path)]

    pool = Pool(os.cpu_count())
    for i, _ in enumerate(pool.imap_unordered(process_file, f_names), 1):
        sys.stderr.write('\rpercentage of files completed: {0:%}'.format(i/len(f_names)))


def process_file(file):
    refactors_list = [  rename_argument, return_optimal, add_arguments, rename_api,
                        rename_local_variable, add_local_variable, rename_method_name, 
                        enhance_for_loop, enhance_filed, enhance_if, add_print]

    filename = os.fsdecode(str(file) + '.java')
    total_file_transformations = 0

    try:
        K = 1
        for transformation in refactors_list:

            r_file = open(f'{source_path}/{filename}', 'r', encoding='ISO-8859-1')
            code = r_file.read()

            new_code = generate_adversarial(K, code, transformation)

            if new_code is not '' and new_code != code:

                new_code = '\n'.join(new_code.split('\n')[1:-1])

                code = '\n'.join(code.split('\n')[1:-1])

                f1 = [line + '\n' for line in code.split('\n')]
                f2 = [line + '\n' for line in new_code.split('\n')]

                delta = difflib.unified_diff(f1, f2, fromfile=filename, tofile=f'{transformation.__name__}_{filename}')

                diffs = ''
                for line in delta:
                    diffs += line
                
                instance = {'status': 'success',
                            'file': filename,
                            'original': code,
                            'transformed': new_code,
                            'transformation': transformation.__name__,
                            'context_diff': diffs
                            }

                if np.random.uniform() < val_frac:
                    valid_set.write(json.dumps(instance) + '\n')
                    valid_set.flush()

                else:
                    training_set.write(json.dumps(instance) + '\n')
                    training_set.flush()

                total_file_transformations += 1
            
            r_file.close()

    except OSError as exc:
        print(f'Error: {exc}')
        sys.exit()


def main(args):
    if args.download_dataset:
        download_dataset(args.language, args.data_path)

    if args.export_functions:
        export_functions(args.language, args.data_path)

    if args.augment_functions:
        augment_functions(args.language, args.data_path, args.validation_frac)


def parse_args():
    parser = argparse.ArgumentParser("generate the dataset for pretraining")
    parser.add_argument('--data_path', type=str, default='resources/pretraining/data', help='data path directory')
    parser.add_argument('--language', type=str, default='java', help='programming language of the dataset')
    parser.add_argument('--download_dataset', type=bool, default=False, help='download the dataset from CodeSearchNet')
    parser.add_argument('--export_functions', type=bool, default=False, help='export the functions from the dataset')
    parser.add_argument('--augment_functions', type=bool, default=False, help='augment the functions from the dataset')
    parser.add_argument('--validation_frac', type=float, default=0.001, help='fraction of the dataset to be used for validation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
