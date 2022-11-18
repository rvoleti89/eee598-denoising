import requests
import pandas as pd
import os
import wget
import tarfile
from pathlib import Path
from csv import QUOTE_NONE

DEFAULT_PATH = os.path.expanduser('~/data/sentences/clean')


def clean_text_regex(df: pd.DataFrame):
    df['sent_1'] = df['sent_1'].str.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ")
    df['sent_2'] = df['sent_2'].str.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ")


def download_and_load_sts_data(url="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz", path=DEFAULT_PATH):
    if not Path(os.path.join(path, 'stsbenchmark')).is_dir():
        if not Path(os.path.join(path, 'Stsbenchmark.tar.gz')).is_file():
            wget.download(url, path)

        with tarfile.open(os.path.join(path, 'Stsbenchmark.tar.gz')) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=path)

        os.remove(os.path.join(path, 'Stsbenchmark.tar.gz'))

    sts_benchmark_path = os.path.join(path, 'stsbenchmark')

    sts_train = pd.read_csv(os.path.join(sts_benchmark_path, 'sts-train.csv'), sep='\t', usecols=[4, 5, 6],
                            names=['sim', 'sent_1', 'sent_2'], quoting=QUOTE_NONE)
    sts_train['set'] = 'sts_b_train'
    clean_text_regex(sts_train)

    sts_dev = pd.read_csv(os.path.join(sts_benchmark_path, 'sts-dev.csv'), sep='\t', usecols=[4, 5, 6],
                          names=['sim', 'sent_1', 'sent_2'], quoting=QUOTE_NONE)
    sts_dev['set'] = 'sts_b_dev'
    clean_text_regex(sts_dev)

    sts_test = pd.read_csv(os.path.join(sts_benchmark_path, 'sts-test.csv'), sep='\t', usecols=[4, 5, 6],
                           names=['sim', 'sent_1', 'sent_2'], quoting=QUOTE_NONE)
    sts_test['set'] = 'sts_b_test'
    clean_text_regex(sts_test)

    return sts_train, sts_dev, sts_test


def download_sick(url, set_name=None):
    response = requests.get(url).text

    lines = response.split("\n")[1:]
    lines = [l.split("\t") for l in lines if len(l) > 0]
    lines = [l for l in lines if len(l) == 5]

    df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
    df['sim'] = pd.to_numeric(df['sim'])
    df['set'] = set_name
    clean_text_regex(df)
    return df


def save_text_files(df: pd.DataFrame, path: str = DEFAULT_PATH):
    file_name = df['set'].iloc[0]
    full_path = os.path.join(path, file_name)
    df['sent_1'].to_csv(full_path + '_sent1.txt', index=False)
    df['sent_2'].to_csv(full_path + '_sent2.txt', index=False)


if __name__ == '__main__':
    sts_train, sts_dev, sts_test = download_and_load_sts_data()
    sick_train = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt",
                               set_name='sick_train')
    sick_dev = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt",
                             set_name='sick_dev')
    sick_test = download_sick(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt",
        set_name='sick_test')

    save_path = DEFAULT_PATH

    try:
        save_text_files(sick_train, path=save_path)
        save_text_files(sick_dev, path=save_path)
        save_text_files(sick_test, path=save_path)
        save_text_files(sts_train, path=save_path)
        save_text_files(sts_dev, path=save_path)
        save_text_files(sts_test, path=save_path)
    except OSError:
        os.makedirs(save_path)
        save_text_files(sick_train, path=save_path)
        save_text_files(sick_dev, path=save_path)
        save_text_files(sick_test, path=save_path)
        save_text_files(sts_train, path=save_path)
        save_text_files(sts_dev, path=save_path)
        save_text_files(sts_test, path=save_path)

    print('Done!')
