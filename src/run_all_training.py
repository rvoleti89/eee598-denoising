# -*- coding: utf-8 -*-

"""Console script for autoencoder training."""

import os
import sys
import importlib
import click
import torch
import logging
import pandas as pd
from pathlib import Path

# Imports from importlib
sent_loader = importlib.import_module('.data.01_load_sentences', 'src')
corrupter = importlib.import_module('.tools.02_corrupt_sentences', 'src')
word_vec_loader = importlib.import_module('.tools.04_load_pretrained_word_vector_models', 'src')
sentence_embedding = importlib.import_module('.tools.05_generate_sentence_embeddings', 'src')
trainer = importlib.import_module('.models.06_train_autoencoder', 'src')

# Global variables
DEVICE = trainer.DEVICE

# Set up logger for run configuration
logger = logging.getLogger(__name__)


@click.command()
@click.option('--wer', 'wer', default=0.20, help='Specified WER for corruption of original text', type=float)
@click.option('--set', 'corpus', default='all', help='String specifying sentence corpus to use.', type=str)
@click.option('--emb', 'embedding', default='infersent', help='String specifying sentence embedding type, currently '
                                                              'supporting either "infersent" or "sif"', type=str)
@click.option('--out', '-o', 'output', default=Path('data/processed'), help='path for directory for saving output model')
@click.option('--epochs', 'num_epochs', default=1000, help='Number of training epochs', type=int)
@click.option('--hidden', '-h', 'num_hidden_units', default=5200, help='Number of units in hidden layer for training',
              type=int)
@click.option('--lr', 'learning_rate', default=0.002, help='Learning rate for pytorch optimizer', type=float)
@click.option('--batch', '-b', 'batch_size', default=512, help='Mini-batch size for training', type=int)
@click.option('--optim', 'optimizer', default='adam', type=str, help="Choice of 'adam', 'sgd', or 'adagrad' "
                                                                     "(default adam)")
@click.option('--loss', 'loss', default='cosine', help='Loss function for training, either "mse" or "cosine"')
@click.option('--cpu', 'bertcpu', default=False, help='boolean to decide if BERT computation should use cpu')
@click.option('--name', 'name', default=None, help='Name for state_dict file for saved model')
@click.option('--large', 'large', default=False, help='Determines whether or not to use BERT-large model')
@click.option('--network', '-n', 'network_type', default='single', help='"single" or "stacked"')
@click.option('--dropout', '-d', 'dpout', default=0.5, help='amount of dropout to apply between layers')
@click.option('--tied', '-t', 'tied', is_flag=True, help='set flag if tied weights are desired')
@click.option('--act', '-a', 'activation', default='leaky', help='Activation function for denoiser, can be "leaky" '
                                                                 '(leaky ReLU), "sigmoid", or "tanh"')
@click.option('--schedule', '-s', 'schedule', is_flag=True, help='set flag to use learning rate scheduler')
def main(wer, corpus, embedding, output, num_epochs, num_hidden_units, learning_rate, batch_size,
         optimizer, loss, bertcpu, name, large, network_type, dpout, tied, activation, schedule):
    # Set optimizer
    if optimizer == 'adagrad':
        optim = torch.optim.Adagrad
    elif optimizer == 'adam':
        optim = torch.optim.Adam
    elif optimizer == 'sgd':
        optim = torch.optim.SGD
    else:
        raise(ValueError, 'Optimizer not supported')

    # Set activation
    if activation == 'leaky':
        activation_fun = torch.nn.LeakyReLU()
    elif activation == 'sigmoid':
        activation_fun = torch.nn.Sigmoid()
    elif activation == 'tanh':
        activation_fun = torch.nn.Tanh()
    else:
        raise(ValueError, 'activation function not supported (must be "leaky", "sigmoid", or "tanh"')

    # Set loss
    if loss == 'cosine':
        loss_fun = trainer.CosineDistLoss()
    elif loss == 'mse':
        loss_fun = torch.nn.MSELoss(reduction='sum')
    else:
        raise(ValueError, 'Unsupported loss function, must be "mse" or "cosine"')

    clean_save_path = os.path.expanduser('~/data/sentences/clean')
    corrupt_save_path = os.path.expanduser('~/data/sentences/corrupt')

    # Steps: download corpus if it doesn't exist, save clean sentences, compute sentence
    # embeddings, and train
    if corpus == 'all':
        print('Loading sentences from SICK and STS-Benchmark...')
        sick_train = sent_loader.download_sick(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt",
            set_name='sick_train')
        sick_dev = sent_loader.download_sick(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt",
            set_name='sick_dev')

        sts_b_train, sts_b_dev, _ = sent_loader.download_and_load_sts_data()

        if not Path(clean_save_path, 'sick_train_sent1.txt').is_file() or \
           not Path(clean_save_path, 'sick_train_sent2.txt').is_file() or \
           not Path(clean_save_path, 'sick_dev_sent1.txt').is_file() or \
           not Path(clean_save_path, 'sick_dev_sent2.txt').is_file() or \
           not Path(clean_save_path, 'sts_b_train_sent1.txt').is_file() or \
           not Path(clean_save_path, 'sts_b_train_sent2.txt').is_file() or \
           not Path(clean_save_path, 'sts_b_dev_sent1.txt').is_file() or \
           not Path(clean_save_path, 'sts_b_dev_sent2.txt').is_file():

            try:
                sent_loader.save_text_files(sick_train, path=clean_save_path)
                sent_loader.save_text_files(sick_dev, path=clean_save_path)
                sent_loader.save_text_files(sts_b_train, path=clean_save_path)
                sent_loader.save_text_files(sts_b_dev, path=clean_save_path)
            except OSError:
                os.makedirs(clean_save_path)
                sent_loader.save_text_files(sick_train, path=clean_save_path)
                sent_loader.save_text_files(sick_dev, path=clean_save_path)
                sent_loader.save_text_files(sts_b_train, path=clean_save_path)
                sent_loader.save_text_files(sts_b_dev, path=clean_save_path)

        # Save corrupt versions to update probs dictionary
        if not Path(corrupt_save_path, 'corrupted_sick_train_sent1.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sick_train_sent2.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sick_dev_sent1.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sick_dev_sent2.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_train_sent1.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_train_sent2.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_dev_sent1.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_dev_sent2.txt').is_file():

            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_train_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_train_sent2.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_dev_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_dev_sent2.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_train_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_train_sent2.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_dev_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_dev_sent2.txt'), wer)

        train_list = list(pd.concat([sick_train['sent_1'], sick_train['sent_2'],
                                     sts_b_train['sent_1'], sts_b_train['sent_2']]))

        dev_list = list(pd.concat([sick_dev['sent_1'], sick_dev['sent_2'],
                                   sts_b_dev['sent_1'], sts_b_dev['sent_2']]))

    elif corpus == 'sick':
        print('Loading sentences from SICK corpus...')
        sick_train = sent_loader.download_sick(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt",
            set_name='sick_train')
        sick_dev = sent_loader.download_sick(
            "https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt",
            set_name='sick_dev')

        train_df = sick_train
        dev_df = sick_dev

        if not Path(clean_save_path, 'sick_train_sent1.txt').is_file() or \
           not Path(clean_save_path, 'sick_train_sent2.txt').is_file() or \
           not Path(clean_save_path, 'sick_dev_sent1.txt').is_file() or \
           not Path(clean_save_path, 'sick_dev_sent2.txt').is_file():

            try:
                sent_loader.save_text_files(sick_train, path=clean_save_path)
                sent_loader.save_text_files(sick_dev, path=clean_save_path)
            except OSError:
                os.makedirs(clean_save_path)
                sent_loader.save_text_files(sick_train, path=clean_save_path)
                sent_loader.save_text_files(sick_dev, path=clean_save_path)

        # Save corrupt versions to update probs dictionary
        if not Path(corrupt_save_path, 'corrupted_sick_train_sent1.txt').is_file() or \
           not Path(corrupt_save_path, 'corrupted_sick_train_sent2.txt').is_file() or \
           not Path(corrupt_save_path, 'corrupted_sick_dev_sent1.txt').is_file() or \
           not Path(corrupt_save_path, 'corrupted_sick_dev_sent2.txt').is_file():

            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_train_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_train_sent2.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_dev_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sick_dev_sent2.txt'), wer)

        train_list = list(pd.concat([train_df['sent_1'], train_df['sent_2']]))
        dev_list = list(pd.concat([dev_df['sent_1'], dev_df['sent_2']]))

    elif corpus == 'sts-b':
        sts_b_train, sts_b_dev, _ = sent_loader.download_and_load_sts_data()

        if not Path(clean_save_path, 'sts_b_train_sent1.txt').is_file() or \
                not Path(clean_save_path, 'sts_b_train_sent2.txt').is_file() or \
                not Path(clean_save_path, 'sts_b_dev_sent1.txt').is_file() or \
                not Path(clean_save_path, 'sts_b_dev_sent2.txt').is_file():
            try:
                sent_loader.save_text_files(sts_b_train, path=clean_save_path)
                sent_loader.save_text_files(sts_b_dev, path=clean_save_path)
            except OSError:
                os.makedirs(clean_save_path)
                sent_loader.save_text_files(sts_b_train, path=clean_save_path)
                sent_loader.save_text_files(sts_b_dev, path=clean_save_path)

            # Save corrupt versions to update probs dictionary
        if not Path(corrupt_save_path, 'corrupted_sts_b_train_sent1.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_train_sent2.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_dev_sent1.txt').is_file() or \
                not Path(corrupt_save_path, 'corrupted_sts_b_dev_sent2.txt').is_file():
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_train_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_train_sent2.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_dev_sent1.txt'), wer)
            corrupter.save_corrupt_sentences(os.path.join(clean_save_path, 'sts_b_dev_sent2.txt'), wer)
        train_list = list(pd.concat([sts_b_train['sent_1'], sts_b_train['sent_2']]))
        dev_list = list(pd.concat([sts_b_dev['sent_1'], sts_b_dev['sent_2']]))
    else:
        raise (ValueError, 'Corpus must be "sick", "sts-b", or "all" for now, will add more corpora in the future')

    # Encode clean sentences with embedding method
    # Only compute output (clean embeddings) once, because it won't change during training, save DataFrame

    train_list = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ") for line in train_list]
    dev_list = [line.replace(r"[^A-Za-z0-9(),!?@\'\_\n]", " ") for line in dev_list]
    if embedding == 'infersent':
        dim = 4096
        h_dim = num_hidden_units
        weights_tensor = torch.empty(h_dim, dim)
        torch.nn.init.eye_(weights_tensor)
        if network_type == 'stacked':
            model = trainer.StackedDenoiser(dim, h_dim, weight=weights_tensor, dropout=dpout)
        elif network_type == 'single':
            model = trainer.SentenceDenoiser(dim, h_dim, weight=weights_tensor,
                                             dropout=dpout, tied=tied, activation=activation_fun)
        else:
            raise(ValueError, 'network_type must be "single" or "stacked"')
        word_emb_model = None
    elif embedding == 'bert':
        if large:
            dim = 1024
        else:
            dim = 768
        h_dim = num_hidden_units
        weights_tensor = torch.empty(h_dim, dim)
        torch.nn.init.eye_(weights_tensor)
        model = trainer.SentenceDenoiser(dim, h_dim, weight=weights_tensor,
                                         dropout=dpout, tied=tied, activation=activation_fun)
        word_emb_model = None
    elif embedding == 'sif' or embedding == 'average':
        word_emb_model = word_vec_loader.load_w2v_gensim_model()
        dim = 300
        h_dim = num_hidden_units
        weights_tensor = torch.empty(h_dim, dim)
        torch.nn.init.eye_(weights_tensor)
        model = trainer.SentenceDenoiser(dim, h_dim, weight=weights_tensor,
                                         dropout=dpout, tie=tied, activation=activation_fun)
    else:
        model = None
        word_emb_model = None

    if tied:
        tied_value = 'tied'
    else:
        tied_value = 'not_tied'

    if schedule:
        schedule_value = 'scheduled'
    else:
        schedule_value = 'not_scheduled'

    if name is None:
        name = f'{embedding}_h{num_hidden_units}_e{num_epochs}_wer{wer*100}_{loss}' \
            f'loss_{corpus}_{network_type}_act_{activation}_{optimizer}_lr{learning_rate}_b{batch_size}' \
            f'_dropout{dpout}_{schedule_value}_{tied_value}.dict'

    trainer.train_denoiser(train_list, dev_list, wer=wer, sent_embedding_type=embedding,
                           model=model, batch_size=batch_size, n_epochs=num_epochs,
                           device=DEVICE, learning_step=learning_rate, criterion=loss_fun,
                           optim=optim, model_dir=output, word_embedding=word_emb_model,
                           bert_cpu=bertcpu, model_name=name, corpus=corpus, tied=tied, schedule=schedule)

    print('Done training!')


if __name__ == '__main__':
    sys.exit(main())  # pragma: no cover
