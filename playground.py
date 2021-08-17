import torch.nn as nn

capa = nn.LSTM(64,256)
print(type(capa))

if isinstance(capa,nn.modules.rnn.LSTM):
    print("yes")
else:
    print("no")