"""
@Author: Andrés Alejandro Córdova Galleguillos
"""

###
# Import libraries
###
from __future__ import unicode_literals, print_function, division

import random

# Import Pytorch
import torch
import torch.nn as nn
from torch import optim

import datetime
import time
import math

# Import functions of file utils
from utils import *

from argparse import ArgumentParser

from networks import EncoderRNN, DecoderRNN, AttnDecoderRNN

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 200

# Use CUDA if it is available
device = "cuda" if torch.cuda.is_available() else "cpu"


def prepareData(lang1, lang2, reverse=False):
    """
    Function
    @param lang1: Language of origin
    @param lang2: Target language
    @param reverse: boolean to reverse the pairs
    @return input_lang:
    @return output_lang:
    @return pairs: a list of pairs, each pair a sentence in the
                   original language, the other of the target language
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words in ", input_lang.name, " : ", input_lang.n_words)
    print("Counted words in ", output_lang.name, " : ", output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('de', 'es', False)
print(random.choice(pairs))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, attention=True, recurrent='GRU'):
    if recurrent == 'LSTM':
        encoder_hidden = encoder.initLSTMHidden()
    else:
        encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def format_time(result):
    date = datetime.utcfromtimestamp(result)
    output = datetime.strftime(date, "%H:%M:%S:%f")
    return output


def trainIters(encoder, decoder, n_iters, print_every=100, learning_rate=0.01, attention=True, recurrent='GRU'):
    print("Training...")
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,attention=attention,recurrent=recurrent)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

    final_time = time.time()
    print('Training time: {}'.format(format_time(final_time - start)))
    return encoder, decoder


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH, recurrent_type ='GRU'):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        if recurrent_type == 'GRU':
            encoder_hidden = encoder.initHidden()
        else:
            encoder_hidden = encoder.initLSTMHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            try:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            except:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=100, recurrent_type = 'GRU'):
    destinations = []
    inferences = []

    for i in range(n):
        pair = random.choice(pairs)
        output_words, attentions = evaluate(encoder, decoder, pair[0], recurrent_type = recurrent_type)
        output_sentence = ' '.join(output_words)

        print('INPUT:', pair[0])
        print('REFER:', pair[1])
        print('HYPOS:', output_sentence)
        print('')

        destinations.append(pair[1])
        inferences.append(output_sentence)
        return destinations, inferences

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="noname", help="Name of the model, to save or to load")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training evaluations")
    parser.add_argument("--decoder", type=str, default="A", choices=["A","B"], help="Type of Decoder. A: Attention, B: Basic")
    parser.add_argument("--recurrent", type=str, default="LSTM", choices=["GRU","LSTM"], help="GRU or LSTM")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()
    epochs = args.epochs
    decoder_type = args.decoder
    recurrent_type = args.recurrent
    model_name = args.model_name
    model_name = "Epochs" + str(epochs) if model_name == "noname" else model_name
    device = args.device
    hidden_size = 256

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size,recurrent_type).to(device)
    if decoder_type == "B":
        decoder1 = DecoderRNN(hidden_size, output_lang.n_words,
                              recurrent_type=recurrent_type).to(device)
        bool_attention = False
    else:
        decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                  recurrent_type=recurrent_type, dropout_p=0.1).to(device)
        bool_attention = True
    encoder1, decoder1 = trainIters(encoder1, decoder1, epochs, print_every=1000,
                                    attention=bool_attention, recurrent= recurrent_type)
    checkpoint_encoder = {'input_size': input_lang.n_words,
                          'hidden_size': hidden_size,
                          'state_dict': encoder1.state_dict()}
    checkpoint_decoder = {'decoder': decoder_type,
                          'hidden_size': hidden_size,
                          'output_size': output_lang.n_words,
                          'dropout': 0.1,
                          'state_dict': decoder1.state_dict()}
    log_dir_encoder = make_prex_logdir("encoder", model_name)
    log_dir_decoder = make_prex_logdir("decoder", model_name)
    last_log_dir_encoder = make_last_logdir("last_encoder", model_name)
    last_log_dir_decoder = make_last_logdir("last_decoder", model_name)

    torch.save(checkpoint_encoder, log_dir_encoder)
    torch.save(checkpoint_decoder, log_dir_decoder)
    empty_last_folder()
    torch.save(checkpoint_encoder, last_log_dir_encoder)
    torch.save(checkpoint_decoder, last_log_dir_decoder)
