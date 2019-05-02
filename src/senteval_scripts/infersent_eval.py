import senteval
import torch
import os
import pickle
import logging
import importlib
import asr_error_simulator as asr
from src.models.models import InferSent
from sklearn import preprocessing

# Word sub DF for ASR Error Simulator, pickle must exist
corrupter = importlib.import_module('.tools.02_corrupt_sentences', 'src')
with open(os.path.expanduser('~/asr_simulator_data/models/word_substitution_df.pkl'), 'rb') as f:
    probs = pickle.load(f)

# Import autoencoder
denoiser = importlib.import_module('.models.06_train_autoencoder', 'src')
inf_denoiser = denoiser.SentenceDenoiser(4096, 5200)
inf_denoiser.load_state_dict(torch.load('data/processed/infersent_h_5200_wer10.0_cosine_loss_all_single_adam_b128_dropout_0.5'))
inf_denoiser.eval()

# Set device if cuda is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inf_denoiser.to(DEVICE)

# Set PATHS
PATH_TO_DATA = os.path.abspath(os.path.join(senteval.__path__[0], '../data'))
PATH_TO_FASTTEXT = os.path.expanduser("~/data/fasttext/crawl-300d-2M.vec")
PATH_TO_GLOVE = os.path.expanduser("~/data/glove/glove.840B.300d.txt")
INF_MODEL_DIR = os.path.expanduser("~/data/infersent")
v = 2  # InferSent Version 2

assert os.path.isdir(INF_MODEL_DIR) and os.path.isfile(PATH_TO_FASTTEXT), 'Set MODEL and FastText PATHs'


# InferSent clean:
def prepare_inf_clean(params, samples):
    samples = [' '.join(s) for s in samples]
    samples = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ").replace("`` ", "\"").replace(" ''", '\"')
               for line in samples]
    params.infersent.build_vocab(samples, tokenize=True)


def batcher_inf_clean(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=True)
    embeddings = preprocessing.normalize(embeddings, norm='l2')
    return embeddings


# InferSent noisy:
def prepare_inf_noisy(params, samples):
    # Add new words to word sub dictionary for replacement:
    if CHECK_NEW_WORDS:
        corrupter.save_corrupt_sentences(samples, wer=WER)


def batcher_inf_noisy(params, batch):
    corrupt_sentences = [asr.replace_error_words(probs, sent, WER) for sent in batch]
    corrupt_sentences = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ").replace("`` ", "\"").replace(" ''", '\"')
                         for line in corrupt_sentences]
    params.infersent.build_vocab(corrupt_sentences, tokenize=True)
    embeddings = params.infersent.encode(corrupt_sentences, bsize=params.batch_size, tokenize=True)
    embeddings = preprocessing.normalize(embeddings, norm='l2')
    return embeddings


# InferSent denoised:
def batcher_inf_denoised(params, batch):
    corrupt_sentences = [asr.replace_error_words(probs, sent, WER) for sent in batch]
    corrupt_sentences = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ").replace("`` ", "\"").replace(" ''", '\"')
                         for line in corrupt_sentences]
    params.infersent.build_vocab(corrupt_sentences, tokenize=True)
    embeddings = params.infersent.encode(corrupt_sentences, bsize=params.batch_size, tokenize=True)
    embeddings = torch.tensor(embeddings).to(DEVICE)
    embeddings = inf_denoiser(embeddings)
    embeddings = embeddings.detach().cpu().numpy()
    embeddings = preprocessing.normalize(embeddings, norm='l2')
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""
# Define senteval params:
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 128,
                                 'tenacity': 5, 'epoch_size': 4}
inf_file = os.path.join(INF_MODEL_DIR, 'infersent%s.pkl' % v)
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': v}

model = InferSent(params_model)
model.load_state_dict(torch.load(inf_file))
model.set_w2v_path(PATH_TO_FASTTEXT)

params_senteval['infersent'] = model.to(DEVICE)
params_senteval['batch_size'] = 256

# Set up logger
logging.basicConfig(filename='data/senteval_logs/infersent_logs_5200_WER25_STSBenchmark.log',
                    format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    WER = 0.10
    CHECK_NEW_WORDS = True
    task_test = ['SICKRelatedness']
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark',
                      'SICKRelatedness', 'SICKEntailment']

    # # InferSent Clean
    # logging.debug('CLEAN RESULTS: \n\n\n')
    # se_inf_clean = senteval.engine.SE(params_senteval, batcher_inf_clean, prepare_inf_clean)
    # inf_clean_results = se_inf_clean.eval(transfer_tasks)
    # print(inf_clean_results)

    # InferSent Noisy
    logging.debug('NOISY RESULTS: \n\n\n')
    se_inf_noisy = senteval.engine.SE(params_senteval, batcher_inf_noisy, prepare_inf_noisy)
    inf_noisy_results = se_inf_noisy.eval(transfer_tasks)
    print(inf_noisy_results)

    # InferSent Denoised
    logging.debug('DE-NOISED RESULTS: \n\n\n')
    se_inf_denoised = senteval.engine.SE(params_senteval, batcher_inf_denoised)
    inf_denoised_results = se_inf_denoised.eval(transfer_tasks)
    print(inf_denoised_results)
