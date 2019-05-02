import os
import subprocess


def save_corrupt_sentences(text_file, wer: float,
                           corrupted_directory: str = os.path.expanduser('~/data/sentences/corrupt')):
    """
    :param text_file: path to text file to corrupt, a string or a list of sentence strings
    :param wer: word error rate, a float variable between 0.0 and 1.0
    :param corrupted_directory: path to directory for corrupted text, a string
    :return: None
    """

    if type(text_file) == str:
        process_call = ['corrupt_text_file', '-f', f'{text_file}', '-e', f'{wer}', '-o', f'{corrupted_directory}']
        subprocess.call(process_call)

    elif type(text_file) == list:
        file_to_corrupt = os.path.join(corrupted_directory, 'text.txt')
        with open(file_to_corrupt, 'wb') as f:
            try:
                text_list = [str.encode(item + '\n') for item in text_file]
            except TypeError:
                text_list = [' '.join(item) for item in text_file]
                text_list = [str.encode(item + '\n') for item in text_list]
            f.writelines(text_list)
        process_call = ['corrupt_text_file', '-f', f'{file_to_corrupt}', '-e', f'{wer}', '-o', f'{corrupted_directory}']
        subprocess.call(process_call)


if __name__ == '__main__':
    path_to_train1 = os.path.expanduser('~/data/sentences/clean/sick_train_sent1.txt')
    path_to_train2 = os.path.expanduser('~/data/sentences/clean/sick_train_sent2.txt')
    save_corrupt_sentences(path_to_train1, 0.20)
    save_corrupt_sentences(path_to_train2, 0.20)

    path_to_dev1 = os.path.expanduser('~/data/sentences/clean/sick_dev_sent1.txt')
    path_to_dev2 = os.path.expanduser('~/data/sentences/clean/sick_dev_sent2.txt')
    save_corrupt_sentences(path_to_dev1, 0.20)
    save_corrupt_sentences(path_to_dev2, 0.20)

    print('Done!')
