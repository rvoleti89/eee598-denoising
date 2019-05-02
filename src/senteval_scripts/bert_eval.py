import senteval
import torch
import os
import pickle
import logging
import importlib
import numpy as np
import asr_error_simulator as asr
from bert_embedding import BertEmbedding
from sklearn import preprocessing
if torch.cuda.is_available():
    import mxnet as mx

# Word sub DF for ASR Error Simulator, pickle must exist
corrupter = importlib.import_module('.tools.02_corrupt_sentences', 'src')
with open(os.path.expanduser('~/asr_simulator_data/models/word_substitution_df.pkl'), 'rb') as f:
    probs = pickle.load(f)

# Import autoencoder
denoiser = importlib.import_module('.models.06_train_autoencoder', 'src')
bert_denoiser = denoiser.SentenceDenoiser(768, 1300)
bert_denoiser.load_state_dict(torch.load('data/processed/best_bert_1300.dict'))
bert_denoiser.eval()

# Set device if cuda is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_denoiser.to(DEVICE)

# Set PATHS
PATH_TO_DATA = os.path.abspath(os.path.join(senteval.__path__[0], '../data'))


# BERT clean:
def prepare_bert_clean(params, samples):
    if torch.cuda.is_available() and params.dev != 'cpu':
        ctx = mx.gpu(0)
        if params.large:
            params.bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_uncased', ctx=ctx)
        else:
            params.bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased', ctx=ctx)
    else:
        if params.large:
            params.bert = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
        else:
            params.bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')


def batcher_bert_clean(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.bert.embedding(sentences, 'avg')
    return np.array([emb[0] for emb in embeddings])


# # InferSent noisy:
# def prepare_inf_noisy(params, samples):
#     # Add new words to word sub dictionary for replacement:
#     if CHECK_NEW_WORDS:
#         corrupter.save_corrupt_sentences(samples, wer=WER)
#
#
# def batcher_inf_noisy(params, batch):
#     corrupt_sentences = [asr.replace_error_words(probs, sent, WER) for sent in batch]
#     params.infersent.build_vocab(corrupt_sentences, tokenize=True)
#     embeddings = params.infersent.encode(corrupt_sentences, bsize=params.batch_size, tokenize=True)
#     embeddings = preprocessing.normalize(embeddings, norm='l2')
#     return embeddings
#
#
# # InferSent denoised:
# def batcher_inf_denoised(params, batch):
#     corrupt_sentences = [asr.replace_error_words(probs, sent, WER) for sent in batch]
#     params.infersent.build_vocab(corrupt_sentences, tokenize=True)
#     embeddings = params.infersent.encode(corrupt_sentences, bsize=params.batch_size, tokenize=True)
#     embeddings = torch.tensor(embeddings).to(DEVICE)
#     embeddings = inf_denoiser(embeddings)[1]
#     embeddings = embeddings.detach().cpu().numpy()
#     embeddings = preprocessing.normalize(embeddings, norm='l2')
#    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""
# Define senteval params:
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 128,
                                 'tenacity': 5, 'epoch_size': 4}

# For BERT small to run on cpu (not enough GPU memory available)
params_senteval['large'] = False
params_senteval['dev'] = 'gpu'
params_senteval['batch_size'] = 32

# Set up logger
logging.basicConfig(filename=None, # 'data/senteval_logs/bert_900_test.log',
                    format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    WER = 0.20
    CHECK_NEW_WORDS = True
    task_test = ['SICKRelatedness']
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark',
                      'SICKRelatedness', 'SICKEntailment']

    # InferSent Clean
    logging.debug('CLEAN RESULTS: \n\n\n')
    se_bert_clean = senteval.engine.SE(params_senteval, batcher_bert_clean, prepare_bert_clean)
    bert_clean_results = se_bert_clean.eval(transfer_tasks)
    print(bert_clean_results)

    # # InferSent Noisy
    # logging.debug('NOISY RESULTS: \n\n\n')
    # se_inf_noisy = senteval.engine.SE(params_senteval, batcher_inf_noisy, prepare_inf_noisy)
    # inf_noisy_results = se_inf_noisy.eval(transfer_tasks)
    # print(inf_noisy_results)
    #
    # # InferSent Denoised
    # logging.debug('DE-NOISED RESULTS: \n\n\n')
    # se_inf_denoised = senteval.engine.SE(params_senteval, batcher_inf_denoised)
    # inf_denoised_results = se_inf_denoised.eval(transfer_tasks)
    # print(inf_denoised_results)
