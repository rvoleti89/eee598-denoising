import csv
import os
import importlib
import nltk
import numpy as np
import pandas as pd
import torch
import wget
from bert_embedding import BertEmbedding
import zipfile
from sklearn.decomposition import TruncatedSVD
from src.models.models import InferSent
from pathlib import Path
if torch.cuda.is_available():
    import mxnet as mx

# Set device if cuda is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define Sentence class
STOP = set(nltk.corpus.stopwords.words("english"))


class Sentence:
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]
        self.tokens_cased = [t for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_cased_without_stop = [t for t in self.tokens_cased if t not in STOP]


#####################################################################################
# Bag-of-Words Average
#####################################################################################
def compute_average_embeddings(sentences, model, cased=False, use_stoplist=False):
    embeddings = []
    for sent in sentences:
        if cased:
            tokens = sent.tokens_cased_without_stop if use_stoplist else sent.tokens_cased
        else:
            tokens = sent.tokens_without_stop if use_stoplist else sent.tokens

        tokens = [token for token in tokens if token in model]
        embedding = np.average([model[token] for token in tokens], axis=0)
        embedding = embedding
        embeddings.append(embedding)
    embeddings = torch.tensor(embeddings)
    return embeddings


#####################################################################################
# Smooth Inverse Frequency (SIF)
#####################################################################################
# Frequencies to use in SIF calculation
PATH_TO_FREQUENCIES_FILE_1 = os.path.abspath("../../data/external/frequencies.tsv")
PATH_TO_DOC_FREQUENCIES_FILE_1 = os.path.abspath("../../data/external/doc_frequencies.tsv")

PATH_TO_FREQUENCIES_FILE_2 = os.path.abspath("data/external/frequencies.tsv")
PATH_TO_DOC_FREQUENCIES_FILE_2 = os.path.abspath("data/external/doc_frequencies.tsv")

# For debugging run_training script:
PATH_TO_FREQUENCIES_FILE_3 = os.path.abspath("../data/external/frequencies.tsv")
PATH_TO_DOC_FREQUENCIES_FILE_3 = os.path.abspath("../data/external/doc_frequencies.tsv")


def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader:
            frequencies[row[0]] = int(row[1])

    return frequencies


def handle_except():
    try:
        word_frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE_2)
        doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE_2)
    except FileNotFoundError:
        word_frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE_3)
        doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE_3)
    return word_frequencies, doc_frequencies


try:
    word_frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE_1)
    doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE_1)
except FileNotFoundError:
    word_frequencies, doc_frequencies = handle_except()

doc_frequencies["NUM_DOCS"] = 1288431


def remove_first_principal_component(x):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(x)
    pc = svd.components_
    xx = x - x @ pc.T * pc
    return xx


def compute_sif_embeddings(sentences, model, cased=False, freqs=word_frequencies, use_stoplist=False, a=0.001):
    """
    :param sentences: list of sentences, each as a Sentence class
    :param model: gensim word vectors model
    :param cased: bool, default True, set to False if all tokens should be made lower-case
    :param freqs: dict, word frequencies downloaded from Wikipedia
    :param use_stoplist: bool, default False, set to True if stop words should be removed
    :param a: float, hyperparameter from SIF implementation for weighted average by frequency
    :return: embeddings, torch tensor of all SIF embeddings
    """
    total_freq = sum(freqs.values())
    embeddings = []
    for sent in sentences:
        if cased:
            tokens = sent.tokens_cased_without_stop if use_stoplist else sent.tokens_cased
        else:
            tokens = sent.tokens_without_stop if use_stoplist else sent.tokens

        tokens = [token for token in tokens if token in model]
        weights = [a/(a+freqs.get(token, 0)/total_freq) for token in tokens]
        try:
            embedding = np.average([model[token] for token in tokens], axis=0, weights=weights)
        except ZeroDivisionError:
            embedding = np.zeros(len(model[tokens[0]]))
        embeddings.append(embedding)
    embeddings = remove_first_principal_component(embeddings)
    embeddings = torch.tensor(embeddings)
    return embeddings


#####################################################################################
# InferSent
#####################################################################################
PATH_TO_FASTTEXT = os.path.expanduser("~/data/fasttext/crawl-300d-2M.vec")
PATH_TO_GLOVE = os.path.expanduser("~/data/glove/glove.840B.300d.txt")
INF_MODEL_DIR = os.path.expanduser("~/data/infersent")


def compute_infersent_embeddings(sentences, vectors='fasttext', download_punkt=False):
    """
    :param sentences: list of 'Sentence' objects to be encoded
    :param vectors: Either 'fasttext' or 'glove' depending on which InferSent model is being used
    :param download_punkt: boolean, set to True if you want to execute nltk.download('punkt')
    :return: embeddings, torch tensor of all infersent embeddings
    """
    if download_punkt:
        nltk.download('punkt')
    if vectors == 'fasttext':
        v = 2
        vec_path = PATH_TO_FASTTEXT
        if not Path(vec_path).is_file():
            try:
                wget.download('https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip',
                              os.path.join(os.path.dirname(vec_path), 'crawl-300d-2M.vec.zip'))
            except OSError:
                os.makedirs(os.path.dirname(vec_path))
                wget.download('https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip',
                              os.path.join(os.path.dirname(vec_path), 'crawl-300d-2M.vec.zip'))

            with zipfile.ZipFile(os.path.join(os.path.dirname(vec_path), 'crawl-300d-2M.vec.zip')) as zip_ref:
                zip_ref.extractall(os.path.dirname(vec_path))
            os.remove(os.path.join(os.path.dirname(vec_path), 'crawl-300d-2M.vec.zip'))

    elif vectors == 'glove':
        v = 1
        vec_path = PATH_TO_GLOVE
        if not Path(vec_path).is_file():
            try:
                wget.download('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                              os.path.join(os.path.dirname(vec_path), 'glove.840B.300d.zip'))
            except OSError:
                os.makedirs(os.path.dirname(vec_path))
                wget.download('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                              os.path.join(os.path.dirname(vec_path), 'glove.840B.300d.zip'))

            with zipfile.ZipFile(os.path.join(os.path.dirname(vec_path), 'glove.840B.300d.zip')) as zip_ref:
                zip_ref.extractall(os.path.dirname(vec_path))
            os.remove(os.path.join(os.path.dirname(vec_path), 'glove.840B.300d.zip'))

    else:
        raise(ValueError, 'vectors must be either "glove" or "fasttext"')

    if not Path(INF_MODEL_DIR).is_dir():
        os.makedirs(INF_MODEL_DIR)

    inf_file = os.path.join(INF_MODEL_DIR, 'infersent%s.pkl' % v)
    if not Path(inf_file).is_file():
        wget.download('https://s3.amazonaws.com/senteval/infersent/infersent%s.pkl' % v, inf_file)

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': v}

    infersent = InferSent(params_model).to(DEVICE)

    infersent.load_state_dict(torch.load(inf_file))

    infersent.set_w2v_path(vec_path)

    raw_sentences = [sent.raw for sent in sentences]

    infersent.build_vocab(raw_sentences, tokenize=True)
    embeddings = infersent.encode(raw_sentences, tokenize=True)
    embeddings = torch.tensor(embeddings)
    return embeddings


#####################################################################################
# BERT
#####################################################################################
def compute_bert_embeddings(sentences, dev=None, large=False):
    if torch.cuda.is_available() and dev != 'cpu':
        ctx = mx.gpu(0)
        if large:
            bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_uncased', ctx=ctx)
        else:
            bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased', ctx=ctx)
    else:
        if large:
            bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
        else:
            bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')

    raw_sentences = [sent.raw for sent in sentences]
    embeddings = bert.embedding(raw_sentences, 'avg')
    embeddings = torch.stack([torch.tensor(emb[0]) for emb in embeddings])
    return embeddings


if __name__ == "__main__":
    load_sents = importlib.import_module('.data.01_load_sentences', 'src')
    sick_test = load_sents.download_sick(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt",
        set_name='sick_train')[:100]

    sick_series = pd.concat([sick_test['sent_1'], sick_test['sent_2']])
    sick_sent_list = list(sick_series.apply(Sentence))

    # Load word2vec model
    load_w2v = importlib.import_module('04_load_pretrained_word_vector_models')
    w2v = load_w2v.load_w2v_gensim_model()

    # sick_sif_embeddings = compute_sif_embeddings(sick_sent_list, w2v, freqs=word_frequencies)
    # sick_inf_embeddings = compute_infersent_embeddings(sick_sent_list)
    sick_bert_embeddings = compute_bert_embeddings(sick_sent_list, dev='cpu')

    print('Done!')
