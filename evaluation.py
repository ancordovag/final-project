import torch
from translation import evaluate, evaluateRandomly
from argparse import ArgumentParser
from utils import get_model, get_last_model
from networks import EncoderRNN, AttnDecoderRNN
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="noname", help="Name of the model, to save or to load")
args = parser.parse_args()
model_name = args.model_name

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
attn_decoder_eval = AttnDecoderRNN(checkpoint_decoder['hidden_size'],
                                   checkpoint_decoder['output_size'], checkpoint_decoder['dropout']).to(device)
attn_decoder_eval.load_state_dict(checkpoint_decoder['state_dict'])
references, candidates = evaluateRandomly(encoder_eval, attn_decoder_eval)
cumm_bleu = 0
N = len(references)
sf = SmoothingFunction()
for reference,candidate in zip(references,candidates):
    ref = reference.split()
    cand = candidate.split()
    print(ref)
    print(cand)
    cumm_bleu += sentence_bleu([ref], cand,smoothing_function=sf.method3)
bleu_score = cumm_bleu/N
print("BLEU score: ", str(bleu_score))