import os
import gensim
import wget
import gzip
import shutil
from pathlib import Path

PATH_TO_WORD2VEC_MODEL = os.path.expanduser("~/data/word2vec/word2vec")
PATH_TO_WORD2VEC = os.path.expanduser("~/data/word2vec/GoogleNews-vectors-negative300.bin")


def load_w2v_gensim_model():
    if not Path(PATH_TO_WORD2VEC_MODEL).is_file():
        directory = os.path.dirname(PATH_TO_WORD2VEC_MODEL)
        if not Path(directory).is_dir():
            print(f'Creating directory at {directory}',
                  ' for saving word2vec pre-trained model')
            os.makedirs(directory)
        if not Path(PATH_TO_WORD2VEC).is_file():
            w2v_archive = os.path.join(directory, 'GoogleNews-vectors-negative300.bin.gz')
            if not Path(w2v_archive).is_file():
                url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
                print(f'Downloading word2vec pre-trained model to {w2v_archive}')
                wget.download(url, os.path.join(directory, 'GoogleNews-vectors-negative300.bin.gz'))
            # Unzip and delete archive
            print(f'Extracting archive to GoogleNews-vectors-negative300.bin')
            with gzip.open(w2v_archive, 'rb') as f_in:
                with open(os.path.join(directory, 'GoogleNews-vectors-negative300.bin'), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f'Deleting {w2v_archive}')
            os.remove(w2v_archive)
        print('Loading and saving gensim word2vec model ...')
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)
        word2vec.save(PATH_TO_WORD2VEC_MODEL)
        print(f'Model saved to {PATH_TO_WORD2VEC_MODEL}')
        w2v_bin = os.path.join(directory, 'GoogleNews-vectors-negative300.bin')
        print(f'Deleting {w2v_bin}')
    else:
        print('Gensim word2vec model found! \nLoading gensim word2vec model...')
        word2vec = gensim.models.KeyedVectors.load(PATH_TO_WORD2VEC_MODEL)
        print('word2vec model loaded!')
    return word2vec


if __name__ == "__main__":
    w2v = load_w2v_gensim_model()
    print('Done!')
