"""
@Author: Andrés Alejandro Córdova Galleguillos
"""

# Import pytorch
import torch

# Import the function of the main script 'translation.py'
from translation import evaluateRandomly

# Import Argument Parser to receive parameters
from argparse import ArgumentParser

# Import the functionalities of utils
from utils import get_model, get_last_model

# Import the models
from networks import EncoderRNN, DecoderRNN, AttnDecoderRNN

# Import all the necessary elements of nltk for evaluation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
import nltk
nltk.download('wordnet')

# Import library to plot
import matplotlib.pyplot as plt

# # Use CUDA if it is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_encoder_decoder(model_name,recurrent):
    log_dir_encoder = get_model("encoder", model_name)
    checkpoint_encoder = torch.load(log_dir_encoder)
    encoder_eval = EncoderRNN(checkpoint_encoder['input_size'],
                              checkpoint_encoder['hidden_size'],
                              recurrent_type=recurrent).to(device)
    encoder_eval.load_state_dict(checkpoint_encoder['state_dict'])

    log_dir_decoder = get_model("decoder", model_name)
    checkpoint_decoder = torch.load(log_dir_decoder)
    decoder_type = checkpoint_decoder['decoder']
    # Depending on the type of the decoder, the parameters of the checkpoint change
    if decoder_type=='A':
        decoder_eval = AttnDecoderRNN(checkpoint_decoder['hidden_size'],
                        checkpoint_decoder['output_size'], recurrent_type=recurrent,
                        dropout_p=checkpoint_decoder['dropout'],
                        ).to(device)
    else:
        decoder_eval = DecoderRNN(checkpoint_decoder['hidden_size'],
                        checkpoint_decoder['output_size'],
                        recurrent_type=recurrent).to(device)
    decoder_eval.load_state_dict(checkpoint_decoder['state_dict'])

    return encoder_eval, decoder_eval

def get_scores(references, candidates):
    sf = SmoothingFunction()
    # Iterate through the reference-candidate pairs and calculate the scores.
    bleus = []
    meteors = []
    for reference,candidate in zip(references,candidates):
        meteor = single_meteor_score(reference,candidate)
        meteors.append(meteor)
        ref = reference.split()
        cand = candidate.split()
        bleu = sentence_bleu([ref], cand, smoothing_function=sf.method3)
        bleus.append(bleu)
    return bleus, meteors

# Call the evaluation function of the translation script with the number of sentences to evaluate
# It returns a list of the original translations, and the predicted by the model
encoder_eval, decoder_eval = get_encoder_decoder("BKGRU","GRU")
references, candidates, lengths = evaluateRandomly(encoder_eval, decoder_eval, n=500, recurrent_type="GRU", display=False)
bleus, meteors = get_scores(references,candidates)
plt.scatter(lengths,meteors, label="Basic GRU",c='b')
encoder_eval_A, decoder_eval_A = get_encoder_decoder("AKGRU","GRU")
references_A, candidates_A, lengths_A = evaluateRandomly(encoder_eval_A, decoder_eval_A, n=500, recurrent_type="GRU", display=False)
bleus_A,meteors_A = get_scores(references_A,candidates_A)
plt.scatter(lengths_A,meteors_A, label="Attention GRU",c='r')

plt.title("METEOR scores vs the lengths of the sentences",fontsize=19)
plt.xlabel("Length of the sentences",fontsize=18)
plt.ylabel("METEOR",fontsize=18)
plt.grid(True)
plt.legend()
plt.show()