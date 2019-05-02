import pandas as pd
import os
import importlib


def load_corrupted_sentences_to_df(sent_df: pd.DataFrame,
                                   corrupted_text_file1: str, corrupted_text_file2: str, wer: float = 0.25):
    """
    :param sent_df: DataFrame containing original sentences from corpus in columns labeled sent_1 and sent_2
    :param corrupted_text_file1: str path to text file with corrupted version of sent_1
    :param corrupted_text_file2: str path to text file with corrupted version of sent_2
    :return: sent_df_with_corrupted_sentences: DataFrame with 2 new columns containing corrupted sentences
    """
    corrupted_sent_1 = pd.read_csv(corrupted_text_file1, sep='\t', header=None)
    corrupted_sent_2 = pd.read_csv(corrupted_text_file2, sep='\t', header=None)
    sent_df_with_corrupted_sentences = sent_df.copy()
    sent_df_with_corrupted_sentences['corrupted_sent_1'] = corrupted_sent_1
    sent_df_with_corrupted_sentences['corrupted_sent_2'] = corrupted_sent_2
    sent_df_with_corrupted_sentences['WER'] = wer
    return sent_df_with_corrupted_sentences


if __name__ == "__main__":
    load_df = importlib.import_module('01_load_sentences')
    sentence_df = load_df.download_sick('https://raw.githubusercontent.com/alvations/stasis'
                                        '/master/SICK-data/SICK_train.txt', set_name='sick_train')
    corrupt_path1 = os.path.expanduser('~/data/sentences/corrupt/corrupted_sick_train_sent1.txt')
    corrupt_path2 = os.path.expanduser('~/data/sentences/corrupt/corrupted_sick_train_sent2.txt')
    sentence_df = load_corrupted_sentences_to_df(sentence_df, corrupt_path1, corrupt_path2)
    print('done!')
