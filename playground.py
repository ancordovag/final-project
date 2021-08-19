import torch
import torch.nn as nn

capa = nn.LSTM(64,256)
print(type(capa))

if isinstance(capa,nn.modules.rnn.LSTM):
    print("yes")
else:
    print("no")

a = torch.rand(1,1,256)
b = torch.rand(1,1,256)
c = torch.rand(1,1,256)
d = (b,c)
e = torch.cat((a[0], d[0][0]), 1)
print(e.size())