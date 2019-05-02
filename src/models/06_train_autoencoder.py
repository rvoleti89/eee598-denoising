import logging
import importlib
import torch
import time
import os
import pickle
import asr_error_simulator as asr
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from pathlib import Path

# Set up logger
training_log = logging.getLogger(__name__)

# Set device if cuda is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import sentence embedding module
sentence_embedding = importlib.import_module('.tools.05_generate_sentence_embeddings', 'src')

# import corrupt text file function;
corrupt = importlib.import_module('.tools.02_corrupt_sentences', 'src')


class AutoEncoderDataSet(Dataset):
    """
    DataSet class for autoencoder training
    """
    def __init__(self, clean_embeddings, corrupt_embeddings):
        """
        :param clean_embeddings: list of tensor sentence embeddings for original clean sentences
        :param corrupt_embeddings: list of tensor sentence embeddings for corresponding corrupted sentences
        """
        super(AutoEncoderDataSet, self).__init__()
        self.corrupt_embeddings = corrupt_embeddings
        self.clean_embeddings = clean_embeddings

    def __len__(self):
        return len(self.corrupt_embeddings)

    def __getitem__(self, idx):
        return self.clean_embeddings[idx], self.corrupt_embeddings[idx]


class SentenceDenoiser(torch.nn.Module):
    def __init__(self, embedding_dim, h_dim, weight=None, dropout=0.4, tied=True, activation=torch.nn.LeakyReLU()):
        """
        :param embedding_dim: size of sentence embedding
        :param h_dim: size of hidden layer
        :param weight: None by default in which weights are not tied. Otherwise,
        weights between hidden and output layers are the transpose of weights
        between the input and hidden layer and weights must be of size
        (h_dim, embedding_dim)
        :param dropout: amount of dropout to apply
        """
        super(SentenceDenoiser, self).__init__()

        self.encoder = torch.nn.Linear(embedding_dim, h_dim)

        self.activation = torch.nn.Sequential(torch.nn.Dropout(dropout), activation)

        self.decoder = torch.nn.Linear(h_dim, embedding_dim)

        # If random weight matrix of size (h_dim, embedding_dim) is defined, used tied weights for training
        if weight is not None and tied:
            self.encoder.weight.data = weight.clone()
            self.decoder.weight.data = self.encoder.weight.data.transpose(0, 1)

    def forward(self, data_in, tied=True):
        encoded_feats = self.encoder(data_in)
        activations = self.activation(encoded_feats)
        if tied:
            self.decoder.weight.data = self.encoder.weight.data.transpose(0, 1)
        reconstructed_output = self.decoder(activations)
        return reconstructed_output


class StackedDenoiser(torch.nn.Module):
    def __init__(self, embedding_dim, h_dim1, h_dim2, h_dim3, weight=None, dropout=0.4, tied=True):
        """
        :param embedding_dim: size of sentence embedding
        :param h_dim: size of hidden layer
        :param weight: None by default in which weights are not tied. Otherwise,
        weights between hidden and output layers are the transpose of weights
        between the input and hidden layer and weights must be of size
        (h_dim, embedding_dim)
        :param dropout: amount of dropout to apply
        """
        super(StackedDenoiser, self).__init__()

        encoder_structure = torch.nn.Sequential(torch.nn.Linear(embedding_dim, h_dim1), torch.nn.LeakyReLU(),
                                                torch.nn.Dropout(dropout), torch.nn.Linear(h_dim1, h_dim2),
                                                torch.nn.LeakyReLU(), torch.nn.Dropout(dropout),
                                                torch.nn.Linear(h_dim2, h_dim3), torch.nn.LeakyReLU(),
                                                torch.nn.Dropout(dropout))

        decoder_structure = torch.nn.Sequential(torch.nn.Linear(h_dim3, h_dim2), torch.nn.LeakyReLU(),
                                                torch.nn.Linear(h_dim2, h_dim1), torch.nn.LeakyReLU(),
                                                torch.nn.Linear(h_dim1, embedding_dim))

        self.encoder = encoder_structure
        self.decoder = decoder_structure

        # If random weight matrix of size (h_dim, embedding_dim) is defined, used tied weights for training
        if weight is not None and tied:
            self.encoder._modules['0'].weight.data = weight.clone()
            self.decoder._modules['4'].weight.data = self.encoder._modules['0'].weight.data.transpose(0, 1)

    def forward(self, data_in):
        encoded_feats = self.encoder(data_in)
        reconstructed_output = self.decoder(encoded_feats)
        return reconstructed_output


class CosineDistLoss(torch.nn.Module):

    def __init__(self):
        super(CosineDistLoss, self).__init__()

    def forward(self, data_in, data_out, avg=True):
        cos = torch.nn.CosineSimilarity(dim=1)
        cos_distances = 1 - cos(data_in, data_out)
        loss = torch.sum(cos_distances)
        if avg:
            loss = loss / len(cos_distances)
        return loss


def convert2sentence(list_of_sentence_strings):
    sentence_list = [sentence_embedding.Sentence(sentence) for sentence in list_of_sentence_strings]
    return sentence_list


def generate_corrupt_data(clean_list, probs, wer, embed_fn, type_emb, bert_cpu=True, word_emb=None):
    print('Generating ASR replacement errors...')
    corrupt_list = [asr.replace_error_words(probs, item, wer) for item in clean_list]

    corrupt_list = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ").replace("`` ", "\"").replace(" ''", '\"')
                    for line in corrupt_list]

    print('Generating embeddings...')
    if type_emb == 'infersent':
        corrupt_embeddings = Variable(embed_fn(convert2sentence(corrupt_list)))
    elif type_emb == 'bert':
        if bert_cpu:
            corrupt_embeddings = Variable(embed_fn(convert2sentence(corrupt_list), dev='cpu'))
        else:
            corrupt_embeddings = Variable(embed_fn(convert2sentence(corrupt_list)))
    else:
        corrupt_embeddings = corrupt_list

    return corrupt_embeddings


def train_denoiser(clean_train_list: list, clean_dev_list: list, wer: float, sent_embedding_type: str,
                   word_embedding, model=None, batch_size=128, n_epochs=1000,
                   device=DEVICE, learning_step=0.001, criterion=torch.nn.MSELoss(reduction='mean'),
                   optim=torch.optim.Adam, model_dir=os.path.abspath('data/processed'), workers=0, bert_cpu=False,
                   model_name=None, corpus='all', n_permutations=50, tied=False, schedule=False):
    """
    :param clean_train_list: list of sentence strings in training set
    :param clean_dev_list: list of sentence strings in dev set
    :param wer: float variable for WER, between 0 and 1
    :param sent_embedding_type: str, currently only supports "sif"
    :param word_embedding: gensim KeyedVectors model
    :param model: pytorch module network
    :param batch_size: int training mini batch size
    :param n_epochs: int number of training epochs
    :param device: pytorch device, cpu or cuda
    :param learning_step: step size, float
    :param criterion: loss function
    :param optim: optimizer from pytorch
    :param model_dir: directory where best trained model state_dict is stored
    :param workers: num_workers for DataLoader
    :param bert_cpu: set to True if you want BERT computation to run on cpu instead of gpu
    :param model_name: name of file for model state_dict
    :param corpus: sts-b, sick, or all
    :param n_permutations: # of permutations for the noisy dataset for training
    :param tied: True if tied weights is desired for autoencoder training, False by default
    :param schedule: set to True to use LR scheduler
    :return: None
    """

    # Regex cleaning
    clean_train_list = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ").replace("`` ", "\"").replace(" ''", '\"')
                        for line in clean_train_list]
    clean_dev_list = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ").replace("`` ", "\"").replace(" ''", '\"')
                      for line in clean_dev_list]

    criterion = criterion.to(device)
    model = model.to(device)
    if model_name is None:
        model_path = os.path.join(model_dir, f'best_{sent_embedding_type}_model_{corpus}.dict')
    else:
        model_path = os.path.join(model_dir, model_name)

    optimizer = optim(model.parameters(), lr=learning_step, weight_decay=1e-3)
    if schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1)
    else:
        scheduler = None

    best_result = 10e7
    best_epoch = 0
    report_interval = 20
    model.train()

    # For evaluation purposes
    cos_loss = CosineDistLoss()

    try:
        with open(os.path.expanduser('~/asr_simulator_data/models/word_substitution_df.pkl'), 'rb') as f:
            probs = pickle.load(f)
    except FileNotFoundError:
        corrupt.save_corrupt_sentences(clean_train_list, wer)
        corrupt.save_corrupt_sentences(clean_dev_list, wer)
        with open(os.path.expanduser('~/asr_simulator_data/models/word_substitution_df.pkl'), 'rb') as f:
            probs = pickle.load(f)

    if sent_embedding_type == 'sif' or sent_embedding_type == 'infersent' or sent_embedding_type == 'average' \
            or sent_embedding_type == 'bert':
        # Load embedding function
        if sent_embedding_type == 'average':
            embed = sentence_embedding.compute_average_embeddings
        elif sent_embedding_type == 'sif':
            embed = sentence_embedding.compute_sif_embeddings
        elif sent_embedding_type == 'infersent':
            embed = sentence_embedding.compute_infersent_embeddings
        elif sent_embedding_type == 'bert':
            embed = sentence_embedding.compute_bert_embeddings
        else:
            embed = None

        # Compute embeddings for CLEAN sentences just once and save, if pickle file doesn't already exist
        if not Path(f'data/interim/{sent_embedding_type}_clean_embeddings_{corpus}.pkl').is_file():
            if sent_embedding_type == 'infersent':
                train_clean_embeddings = Variable(embed(convert2sentence(clean_train_list)))
                dev_clean_embeddings = Variable(embed(convert2sentence(clean_dev_list)))
                with open(os.path.abspath(f'data/interim/{sent_embedding_type}_clean_embeddings_{corpus}.pkl'),
                          'wb') as f:
                    torch.save([train_clean_embeddings, dev_clean_embeddings], f)
            elif sent_embedding_type == 'bert':
                if bert_cpu:
                    train_clean_embeddings = Variable(embed(convert2sentence(clean_train_list), dev='cpu'))
                    dev_clean_embeddings = Variable(embed(convert2sentence(clean_dev_list), dev='cpu'))
                else:
                    train_clean_embeddings = Variable(embed(convert2sentence(clean_train_list)))
                    dev_clean_embeddings = Variable(embed(convert2sentence(clean_dev_list)))
                with open(os.path.abspath(f'data/interim/{sent_embedding_type}_clean_embeddings_{corpus}.pkl'),
                          'wb') as f:
                    torch.save([train_clean_embeddings, dev_clean_embeddings], f)
            else:
                train_clean_embeddings = clean_train_list
                dev_clean_embeddings = clean_dev_list
        else:  # Load pickle
            print(f'{sent_embedding_type} embeddings for clean sentences found for "{corpus}" corpus! Loading...')
            with open(f'data/interim/{sent_embedding_type}_clean_embeddings_{corpus}.pkl', 'rb') as f:
                train_clean_embeddings, dev_clean_embeddings = torch.load(f)

        # Based on n_permutations, generate several permutations of training embeddings based on different ASR errors
        # Only if pickle doesn't already exist

        # dev
        file_path = os.path.abspath(f'data/interim/{sent_embedding_type}_dev_corrupt_'
                                    f'embeddings_{corpus}_wer_{wer*100}.pkl')
        if not Path(file_path).is_file():
            dev_corrupt_embeddings = generate_corrupt_data(clean_dev_list, probs, wer, embed, sent_embedding_type)
            with open(file_path, 'wb') as f:
                torch.save(dev_corrupt_embeddings, f)
                print(f'Generated corrupt dev set embeddings and saved pickle file at {file_path}')
        else:
            with open(file_path, 'rb') as f:
                dev_corrupt_embeddings = torch.load(f)
            print(f'Loaded pickle from {file_path} for corrupt dev embeddings')

        dev_set = AutoEncoderDataSet(dev_clean_embeddings, dev_corrupt_embeddings)

        # train
        for n in range(n_permutations):
            file_path = os.path.abspath(f'data/interim/{sent_embedding_type}_'
                                        f'train_corrupt_embeddings_{corpus}_wer_{wer*100}_{n}.pkl')
            if not Path(file_path).is_file():
                train_corrupt_embeddings = generate_corrupt_data(clean_train_list, probs, wer,
                                                                 embed, sent_embedding_type)
                with open(file_path, 'wb') as f:
                    torch.save(train_corrupt_embeddings, f)
                print(f'Generated corrupt train set embeddings and saved pickle file at {file_path}')
            else:
                continue

        print('Start training...')
        for epoch in range(n_epochs):
            alpha = 1.0  # how much to weight the criterion if not cosine
            start_time = time.time()

            file_n = epoch % n_permutations

            file_path = os.path.abspath(
                f'data/interim/{sent_embedding_type}_train_corrupt_'
                f'embeddings_{corpus}_wer_{wer*100}_{file_n}.pkl')
            with open(file_path, 'rb') as f:
                train_corrupt_embeddings = torch.load(f)
            print(f'Loaded pickle from {file_path} for corrupt train embeddings')

            train_set = AutoEncoderDataSet(train_clean_embeddings, train_corrupt_embeddings)

            # Create DataLoaders for train and dev sets
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, num_workers=workers)
            dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True,
                                    pin_memory=True, num_workers=workers)
            accumulated_loss = 0
            accumulated_original_loss = 0
            print('--' * 20)
            training_samples_scanned = 0

            total_cosine_distance_train = 0
            total_orig_cosine_dist_train = 0
            for ind, batch in enumerate(train_loader):
                if sent_embedding_type == 'sif' or sent_embedding_type == 'average':
                    clean_sentences = Variable(embed(
                        convert2sentence(batch[0]), model=word_embedding
                    )).to(device).float()
                    corrupt_sentences = Variable(embed(
                        convert2sentence(batch[1]), model=word_embedding
                    )).to(device).float()
                elif sent_embedding_type == 'infersent' or sent_embedding_type == 'bert':
                    clean_sentences = batch[0].to(device).float()
                    corrupt_sentences = batch[1].to(device).float()
                else:
                    clean_sentences = None
                    corrupt_sentences = None

                training_samples_scanned += len(clean_sentences)
                optimizer.zero_grad()

                output = model(corrupt_sentences, tied=tied)

                # # Force l2-norm preservation
                # norms_to_preserve = torch.norm(corrupt_sentences, p=2, dim=1).view(-1, 1)
                # output = torch.mul(norms_to_preserve, torch.nn.functional.normalize(output))

                if isinstance(criterion, CosineDistLoss):
                    loss = criterion(output, clean_sentences)
                    orig_loss = criterion(corrupt_sentences, clean_sentences)
                else:
                    loss = alpha * criterion(output, clean_sentences) + (1.0 - alpha) * \
                           cos_loss(output, clean_sentences)
                    orig_loss = alpha * criterion(corrupt_sentences, clean_sentences) + (1.0 - alpha) * \
                                cos_loss(corrupt_sentences, clean_sentences)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                accumulated_loss += loss.data.item()
                accumulated_original_loss += orig_loss.data.item()

                # Compute cosine distance if using a different loss function
                if not isinstance(criterion, CosineDistLoss):
                    cosine_dist_train = cos_loss(output, clean_sentences)
                    original_dist_train = cos_loss(corrupt_sentences, clean_sentences)
                    total_cosine_distance_train += torch.sum(cosine_dist_train)
                    total_orig_cosine_dist_train += torch.sum(original_dist_train)
                else:
                    original_dist_train = cos_loss(corrupt_sentences, clean_sentences)
                    total_cosine_distance_train = accumulated_loss
                    total_orig_cosine_dist_train += torch.sum(original_dist_train)

                if ind % report_interval == 0:
                    msg = f'{epoch} completed epochs, {ind} batches'
                    msg += f'\t train batch loss (/sample): {accumulated_loss / training_samples_scanned}'
                    print(msg)

            msg = f'{epoch} completed epochs, {ind} batches'
            msg += f'\t train batch loss (/sample): {accumulated_loss / training_samples_scanned}'
            print(msg)

            # After each epoch, check dev accuracy
            model.eval()
            accumulated_loss = 0
            accumulated_original_loss = 0
            total_cosine_distance_dev = 0
            total_orig_cosine_dist_dev = 0
            for ind, batch in enumerate(dev_loader):
                if sent_embedding_type == 'sif' or sent_embedding_type == 'average':
                    clean_sentences = Variable(embed(
                        convert2sentence(batch[0]), model=word_embedding
                    )).to(device).float()
                    corrupt_sentences = Variable(embed(
                        convert2sentence(batch[1]), model=word_embedding
                    )).to(device).float()
                elif sent_embedding_type == 'infersent' or sent_embedding_type == 'bert':
                    clean_sentences = batch[0].to(device).float()
                    corrupt_sentences = batch[1].to(device).float()
                else:
                    clean_sentences = None
                    corrupt_sentences = None

                output = model(corrupt_sentences, tied=tied)
                if isinstance(criterion, CosineDistLoss):
                    loss = criterion(output, clean_sentences)
                    orig_loss = criterion(corrupt_sentences, clean_sentences)
                else:
                    loss = alpha * criterion(output, clean_sentences) + (1.0 - alpha) * \
                           cos_loss(output, clean_sentences)
                    orig_loss = alpha * criterion(corrupt_sentences, clean_sentences) + (1.0 - alpha) * \
                                cos_loss(corrupt_sentences, clean_sentences)

                accumulated_loss += loss.data.item()
                accumulated_original_loss += orig_loss.data.item()

                # Compute cosine distance if using a different loss function
                if not isinstance(criterion, CosineDistLoss):
                    cosine_dist_dev = cos_loss(output, clean_sentences)
                    total_cosine_distance_dev += torch.sum(cosine_dist_dev)
                    original_dist_dev = cos_loss(corrupt_sentences, clean_sentences)
                    total_orig_cosine_dist_dev += torch.sum(original_dist_dev)
                else:
                    total_cosine_distance_dev = accumulated_loss
                    original_dist_dev = cos_loss(corrupt_sentences, clean_sentences)
                    total_orig_cosine_dist_dev += torch.sum(original_dist_dev)

            # Adjust learning rate on plateau in normalized dev set loss
            if schedule:
                scheduler.step(accumulated_loss / accumulated_original_loss)

            msg = f'{epoch} completed epochs, {ind} batches'
            msg += f'\t dev loss: {accumulated_loss}'
            print(msg)
            elapsed_time = time.time() - start_time

            print(f'Total train set cosine distance: {total_cosine_distance_train}')
            print(f'Total cosine distance of noisy sentences to targets in train: {total_orig_cosine_dist_train}')
            print(f'Train CosDist / Original = {total_cosine_distance_train / total_orig_cosine_dist_train}\n')
            print(f'Total dev set cosine distance: {total_cosine_distance_dev}')
            print(f'Total cosine distance of noisy sentences to targets in dev: {total_orig_cosine_dist_dev}')
            print(f'Dev CosDist / Original = {total_cosine_distance_dev / total_orig_cosine_dist_dev}\n')
            print(f'Epoch {epoch} finished within {timedelta(seconds=elapsed_time)}')

            current_lr = None
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            print(f'Current LR: {current_lr}\n')

            if accumulated_loss / accumulated_original_loss < best_result:
                best_result = accumulated_loss / accumulated_original_loss
                best_epoch = epoch
                print(f'NEW best dev loss/ original: {best_result}\n')
                with open(model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
            print(f'Best dev set result on epoch {best_epoch},\t best dev loss: {best_result}')

        print(f'{epoch} completed epochs of training,\t final dev loss: {accumulated_loss}')
        print(f'Final best result on epoch {best_epoch}: {best_result}')
        print(f'Model state dictionary saved at {model_path}')

    else:
        raise(ValueError, "Sentence embedding type must be 'average', 'sif', or 'infersent' for now.")


def decode_strings(string_list):
    sentence_list = [string.decode() for string in string_list]
    return sentence_list


if __name__ == '__main__':
    clean_path = os.path.expanduser('~/data/sentences/clean')
    corrupt_path = os.path.expanduser('~/data/sentences/corrupt')

    with open(os.path.join(clean_path, 'sick_train_sent1.txt'), 'rb') as f1, open(
            os.path.join(clean_path, 'sick_train_sent2.txt'), 'rb') as f2, open(
            os.path.join(clean_path, 'sick_dev_sent1.txt'), 'rb') as f5, open(
            os.path.join(clean_path, 'sick_dev_sent2.txt'), 'rb') as f6:

        train_list = decode_strings(f1.read().splitlines() + f2.read().splitlines())
        dev_list = decode_strings(f5.read().splitlines() + f6.read().splitlines())

    denoiser = SentenceDenoiser(768, 900, weight=torch.randn(900, 768))

    # # Import word2vec gensim model
    # w2v_import = importlib.import_module('.tools.04_load_pretrained_word_vector_models', 'src')
    # w2v = w2v_import.load_w2v_gensim_model()

    train_denoiser(train_list, dev_list, 0.20, 'infersent', model=denoiser, n_epochs=1000, word_embedding=None,
                   criterion=torch.nn.MSELoss(), batch_size=128, bert_cpu=True, corpus='sick')

    print('done!')
