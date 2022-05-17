
import sys
import torch 
import numpy as np 

print("User Current Version:-", sys.version)
print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device", device)
print(torch.cuda.device_count())

a = np.ones((2,3))

print(a)
tmp = torch.from_numpy(a).type(torch.FloatTensor).to(device)

print(tmp)