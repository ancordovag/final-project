import torch
from translation import evaluateRandomly
from argparse import ArgumentParser
from utils import get_model, get_last_model
from networks import EncoderRNN, DecoderRNN, AttnDecoderRNN
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
import nltk
nltk.download('wordnet')

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="noname", help="Name of the model, to save or to load")
parser.add_argument("--decoder", type=str, default="A", help="Type of Decoder. A: Attention, B: Basic")
parser.add_argument("--sentences", type=int, default="100", help="Number of Sentences to evaluate")
parser.add_argument("--recurrent", type=str, default="GRU", choices=["GRU","LSTM"], help="GRU or LSTM")
args = parser.parse_args()
model_name = args.model_name
to_evaluate = args.sentences

if model_name == "noname":
    log_dir_encoder = get_last_model("encoder")
else:
    log_dir_encoder = get_model("encoder", model_name)
checkpoint_encoder = torch.load(log_dir_encoder)
encoder_eval = EncoderRNN(checkpoint_encoder['input_size'],
                          checkpoint_encoder['hidden_size']).to(device)
encoder_eval.load_state_dict(checkpoint_encoder['state_dict'])

if model_name == "noname":
    log_dir_decoder = get_last_model("decoder")
else:
    log_dir_decoder = get_model("decoder", model_name)
checkpoint_decoder = torch.load(log_dir_decoder)
decoder_type = checkpoint_decoder['decoder']
if decoder_type=='A':
    decoder_eval = AttnDecoderRNN(checkpoint_decoder['hidden_size'],
                    checkpoint_decoder['output_size'], checkpoint_decoder['dropout']).to(device)
else:
    decoder_eval = DecoderRNN(checkpoint_decoder['hidden_size'],
                    checkpoint_decoder['output_size']).to(device)
decoder_eval.load_state_dict(checkpoint_decoder['state_dict'])
references, candidates = evaluateRandomly(encoder_eval, decoder_eval, n=to_evaluate)
cumm_bleu = 0
cumm_meteor = 0
N = len(references)
sf = SmoothingFunction()
for reference,candidate in zip(references,candidates):
    cumm_meteor += single_meteor_score(reference,candidate)
    ref = reference.split()
    cand = candidate.split()
    print(ref)
    print(cand)
    cumm_bleu += sentence_bleu([ref], cand,smoothing_function=sf.method3)
bleu_score = cumm_bleu/N
meteor_score = cumm_meteor/N
print("BLEU score: ", str(bleu_score))
print("METEOR score: ", str(meteor_score))