import pandas as pd
import numpy as np
import functools as ft
import asr_error_simulator as asr
import importlib
import pickle
import scipy
import torch
import os
import math
from src.models.models import InferSent
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from pathlib import Path

# imports from importlib
sentence_embedding = importlib.import_module('.tools.05_generate_sentence_embeddings', 'src')
sent_loader = importlib.import_module('.data.01_load_sentences', 'src')
trainer = importlib.import_module('.models.06_train_autoencoder', 'src')
w2v_loader = importlib.import_module('.tools.04_load_pretrained_word_vector_models', 'src')
w2v = w2v_loader.load_w2v_gensim_model()

with open(os.path.expanduser('~/asr_simulator_data/models/word_substitution_df.pkl'), 'rb') as f:
    probs = pickle.load(f)


# Get corrupt sentence DataFrame
def corrupt_sentences_df(df: pd.DataFrame, wer: float = 0.20, probabilities: pd.DataFrame = probs):
    """
    :param df: DataFrame with columns 'sent_1', 'sent_2', 'sim'
    :param wer: Word Error Rate (WER), default 20%
    :param probabilities: probability substitution DataFrame
    :return: corrupt_df = DataFrame with columns 'sent_1', 'sent_2', 'sim' where sentences have been corrupted
    """
    df2 = df.copy()
    corrupted_1 = df2['sent_1'].apply(lambda x: asr.replace_error_words(probabilities, x, wer))
    corrupted_2 = df2['sent_2'].apply(lambda x: asr.replace_error_words(probabilities, x, wer))
    corrupted_df = pd.DataFrame({'sent_1': corrupted_1, 'sent_2': corrupted_2, 'sim': df2['sim']})
    return corrupted_df


#####################################################################################
# Average
#####################################################################################
def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None, n_repeat=1,
                      spherical=False, cased=False):
    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]

    sims = []
    for n in range(n_repeat):
        for (sent1, sent2) in zip(sentences1, sentences2):
            if cased:
                tokens1 = sent1.tokens_cased_without_stop if use_stoplist else sent1.tokens_cased
                tokens2 = sent2.tokens_cased_without_stop if use_stoplist else sent2.tokens_cased
            else:
                tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
                tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

            tokens1 = [token for token in tokens1 if token in model]
            tokens2 = [token for token in tokens2 if token in model]

            if len(tokens1) == 0 or len(tokens2) == 0:
                sims.append(0)
                continue

            tokfreqs1 = Counter(tokens1)
            tokfreqs2 = Counter(tokens2)

            weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                        for token in tokfreqs1] if doc_freqs else None
            weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                        for token in tokfreqs2] if doc_freqs else None

            if spherical:
                embedding1 = np.average([np.divide(model[token], np.linalg.norm(model[token]))
                                         for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
                embedding2 = np.average([np.divide(model[token], np.linalg.norm(model[token]))
                                         for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

            else:
                embedding1 = np.average([model[token]
                                         for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
                embedding2 = np.average([model[token]
                                         for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

            sim = cosine_similarity(embedding1, embedding2)[0][0]
            sims.append(sim)
    return sims


#####################################################################################
# InferSent
#####################################################################################
PATH_TO_FASTTEXT = os.path.expanduser("~/data/fasttext/crawl-300d-2M.vec")
PATH_TO_GLOVE = os.path.expanduser("~/data/glove/glove.840B.300d.txt")
INF_MODEL_DIR = os.path.expanduser("~/data/infersent")


def run_inf_benchmark(sentences1, sentences2, vectors='fasttext'):
    if vectors == 'fasttext':
        v = 2
        vec_path = PATH_TO_FASTTEXT
    elif vectors == 'glove':
        v = 1
        vec_path = PATH_TO_GLOVE
    else:
        raise(ValueError, 'vectors must be fasttext or glove')

    if not Path(INF_MODEL_DIR).is_dir():
        os.makedirs(INF_MODEL_DIR)

    inf_file = os.path.join(INF_MODEL_DIR, 'infersent%s.pkl' %v)
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': v}

    infersent = InferSent(params_model)

    infersent.load_state_dict(torch.load(inf_file))

    infersent.set_w2v_path(vec_path)

    raw_sentences1 = [sent1.raw for sent1 in sentences1]
    raw_sentences2 = [sent2.raw for sent2 in sentences2]

    infersent.build_vocab(raw_sentences1 + raw_sentences2, tokenize=True)
    embeddings1 = infersent.encode(raw_sentences1, tokenize=True)
    embeddings2 = infersent.encode(raw_sentences2, tokenize=True)

    inf_sims = []
    for (emb1, emb2) in zip(embeddings1, embeddings2):
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        inf_sims.append(sim)

    inf_sims = torch.tensor(inf_sims)
    return inf_sims


#####################################################################################
# Run experiment
#####################################################################################
def run_sts_experiment(df, benchmarks):
    df2 = df.copy()
    sentences1 = [sentence_embedding.Sentence(s) for s in df2['sent_1']]
    sentences2 = [sentence_embedding.Sentence(s) for s in df2['sent_2']]

    pearson_cors, spearman_cors = [], []
    for label, method, _ in benchmarks:
        sims = method(sentences1, sentences2)
        sims = np.array(sims, dtype=np.float)

        # Find indices which are not NaN, only consider those
        valid_idx = np.where(~np.isnan(sims))

        pearson_correlation = np.abs(scipy.stats.pearsonr(sims[valid_idx], df2['sim'].iloc[valid_idx])[0])
        print('P:', label, pearson_correlation)
        pearson_cors.append(pearson_correlation)
        spearman_correlation = np.abs(scipy.stats.spearmanr(sims[valid_idx], df2['sim'].iloc[valid_idx])[0])
        print('S:', label, spearman_correlation, '\n')
        spearman_cors.append(spearman_correlation)

    return pearson_cors, spearman_cors


def run_sts_denoised_experiment(df, benchmarks):
    df2 = df.copy()
    sent1_list = [sentence_embedding.Sentence(sent) for sent in df2['sent_1'].tolist()]
    sent2_list = [sentence_embedding.Sentence(sent) for sent in df2['sent_2'].tolist()]

    pearson_cors, spearman_cors = [], []
    for label, method, model in benchmarks:
        if label == 'INF-FT':
            sent1_embeddings = sentence_embedding.compute_infersent_embeddings(sent1_list)
            sent2_embeddings = sentence_embedding.compute_infersent_embeddings(sent2_list)
            denoised_sent1_embeddings = model(sent1_embeddings)[1]
            denoised_sent2_embeddings = model(sent2_embeddings)[1]

        elif label == 'AVG-W2V':
            sent1_embeddings = sentence_embedding.compute_average_embeddings(sent1_list, model=w2v)
            sent2_embeddings = sentence_embedding.compute_average_embeddings(sent2_list, model=w2v)
            denoised_sent1_embeddings = model(sent1_embeddings)[1]
            denoised_sent2_embeddings = model(sent2_embeddings)[1]

        else:
            raise(ValueError, 'invalid label in benchmark')

        sims = []
        for emb1, emb2 in zip(denoised_sent1_embeddings, denoised_sent2_embeddings):
            emb1 = emb1.detach().numpy()
            emb2 = emb2.detach().numpy()
            sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
            sims.append(sim)

        sims = np.array(sims, dtype=np.float)
        # Find indices which are not NaN, only consider those
        valid_idx = np.where(~np.isnan(sims))

        pearson_correlation = np.abs(scipy.stats.pearsonr(sims[valid_idx], df2['sim'].iloc[valid_idx])[0])
        print('P:', label, pearson_correlation)
        pearson_cors.append(pearson_correlation)
        spearman_correlation = np.abs(scipy.stats.spearmanr(sims[valid_idx], df2['sim'].iloc[valid_idx])[0])
        print('S:', label, spearman_correlation, '\n')
        spearman_cors.append(spearman_correlation)

    return pearson_cors, spearman_cors


if __name__ == '__main__':
    sick_test = sent_loader.download_sick(
        "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt",
        set_name='sick_test')

    sick_test_20 = corrupt_sentences_df(sick_test, wer=0.20)

    # Load pre-trained denoiser and test
    inf_denoiser = trainer.SentenceDenoiser(4096, 5200)
    inf_denoiser.load_state_dict(torch.load('data/processed/best_infersent_model_test.dict'))
    inf_denoiser.eval()

    # Load pre-trained average denoiser
    avg_denoiser = trainer.SentenceDenoiser(300, 300)
    avg_denoiser.load_state_dict(torch.load('data/processed/best_average_model_test.dict'))

    benchmarks = [("INF-FT", ft.partial(run_inf_benchmark, vectors='fasttext'), inf_denoiser),
                  ("AVG-W2V", ft.partial(run_avg_benchmark, model=w2v), avg_denoiser)
                  ]

    pearson_results, spearman_results = {}, {}
    pearson_results['CLEAN'], spearman_results['CLEAN'] = run_sts_experiment(sick_test, benchmarks)
    pearson_results['WER_20'], spearman_results['WER_20'] = run_sts_experiment(sick_test_20, benchmarks)
    pearson_results['DENOISED'], spearman_results['DENOISED'] = run_sts_denoised_experiment(sick_test_20, benchmarks)

    print('cool')
