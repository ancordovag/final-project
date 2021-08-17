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

# # Use CUDA if it is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Receive all the parameters than the user wants to give. Those are:
parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="noname", help="Name of the model, to save or to load")
parser.add_argument("--decoder", type=str, default="B", help="Type of Decoder. A: Attention, B: Basic")
parser.add_argument("--sentences", type=int, default="10", help="Number of Sentences to evaluate")
parser.add_argument("--recurrent", type=str, default="GRU", choices=["GRU","LSTM"], help="GRU or LSTM")
args = parser.parse_args()
model_name = args.model_name
to_evaluate = args.sentences
recurrent = args.recurrent

# if no name is given, then just evaluate the last model
# in any case, initialize the Encoder and the Decoder with the loaded checkpoint
if model_name == "noname":
    log_dir_encoder = get_last_model("encoder")
else:
    log_dir_encoder = get_model("encoder", model_name)
checkpoint_encoder = torch.load(log_dir_encoder)
encoder_eval = EncoderRNN(checkpoint_encoder['input_size'],
                          checkpoint_encoder['hidden_size'],
                          recurrent_type=recurrent).to(device)
encoder_eval.load_state_dict(checkpoint_encoder['state_dict'])

if model_name == "noname":
    log_dir_decoder = get_last_model("decoder")
else:
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

# Call the evaluation function of the translation script with the number of sentences to evaluate
# It returns a list of the original translations, and the predicted by the model
references, candidates = evaluateRandomly(encoder_eval, decoder_eval, n=to_evaluate)

# Initialize the BLEU and the METEOR score in 0.
cumm_bleu = 0
cumm_meteor = 0
N = len(references)
sf = SmoothingFunction()

# Iterate through the reference-candidate pairs and calculate the scores.
for reference,candidate in zip(references,candidates):
    cumm_meteor += single_meteor_score(reference,candidate)
    ref = reference.split()
    cand = candidate.split()
    print(ref)
    print(cand)
    cumm_bleu += sentence_bleu([ref], cand,smoothing_function=sf.method3())
# Calculate the averages
bleu_score = cumm_bleu/N
meteor_score = cumm_meteor/N

# Print results
print("BLEU score: ", str(bleu_score))
print("METEOR score: ", str(meteor_score))