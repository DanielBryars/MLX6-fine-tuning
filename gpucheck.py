import torch
print(torch.__version__)
print(torch.version.cuda)           # Should not be None
print(torch.cuda.is_available()) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (f"device:{device}")