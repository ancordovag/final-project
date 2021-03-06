"""
@Author: Andrés Alejandro Córdova Galleguillos
"""

# Import some useful libraries
from io import open
import unicodedata
from datetime import datetime
import re
import os
import shutil

class Lang:
    """
    Each language is initialized and stored with a name, 3 dictionaries and a word counter.
    The dictionaries are:
    word2index: {word -> index}
    word2count: {word -> number of times the word have been seen}
    index2word: {index -> word}
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        """
        @param: sentence
        Split the sentence in tokens, and pass the tokens to the function addWord.
        This way, all the words of the sentece are added to the object Lang.
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
        @param: word
        If the word has not been added to the Lang, this function add it, assigning an id,
        setting a counter with value 1 for this word, and for the id created the word is assigned.
        If the word existed, just add 1 to the counter.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to
    https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters
    @param s: a string
    @return s: a normalized string
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1="de", lang2="es", reverse=False):
    """
    Read the file and split into lines
    """
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def make_prex_logdir(prex:str, model_name: str):
    """Create unique path to save checkpoints, e.g. models/encoder_Jul22_19-45-59_name"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'models', prex + '_' + current_time + '_' + model_name)
    return logdir

def make_last_logdir(prex:str, model_name: str):
    """Create unique path to save checkpoints, e.g. models/last/encoder_Jul22_19-45-59_name"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'models', 'last', prex + '_' + current_time + '_' + model_name)
    return logdir

def get_model(prex:str,model_name:str):
    """Get the last model saved that matches the prefix and the model_name"""
    full_list = os.listdir("models")
    final_list = [file for file in full_list if prex in file and model_name in file]
    logdir = os.path.join(
        'models', final_list[-1])
    return logdir

def get_last_model(prex:str):
    """Get the last model saved that matches the prefix"""
    full_list = os.listdir(os.path.join('models', 'last'))
    final_list = [file for file in full_list if prex in file]
    logdir = os.path.join(
        'models', 'last', final_list[-1])
    return logdir

def empty_last_folder():
    """Delete all the models saved in the directory 'models/last/' so the last version can replace them"""
    folder = os.path.join('models', 'last')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))