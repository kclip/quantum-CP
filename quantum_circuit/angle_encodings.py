import torch
import torch.nn as nn

### general angle encoding that maps input x to angles used for unitaries in PQC (see Fig. 9 in the paper)

class Angle_Encoding(nn.Module):
    def __init__(self, num_layer_PQC, num_qubits):
        super(Angle_Encoding, self).__init__()
        self.fc1 = nn.Linear(1, 10, bias=True)
        self.fc2 = nn.Linear(10, 10, bias=True)
        self.fc3 = nn.Linear(10, num_layer_PQC*num_qubits*3, bias=True)
        self.activ = torch.nn.ELU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x) 
        x = self.activ(x)
        x = self.fc3(x)
        return x # this is all angles for unitary gates in PQC

class Angle_Encoding_linear(nn.Module):
    # only for input size 1 here
    # please note that usual learned linear angle encoding as in data-reuploading paper considers elementwise product rather than matrix product which nn.Linear does
    def __init__(self, num_layer_PQC, num_qubits):
        super(Angle_Encoding_linear, self).__init__()
        self.fc = nn.Linear(1, num_layer_PQC*num_qubits*3, bias=True) # for input with size 1, exactly same as in the data-reuploading paper
    def forward(self, x):
        x = self.fc(x)
        return x # this is all angles for unitary gates in PQC

class Angle_Encoding_linear_without_weight(nn.Module): ## this is more pure angle encoding used with Fourier analysis ..
    def __init__(self, num_layer_PQC, num_qubits):
        # only bias is trainable here
        super(Angle_Encoding_linear_without_weight, self).__init__()
        self.fc = nn.Linear(1, num_layer_PQC*num_qubits*3, bias=True) 
        with torch.no_grad():
            self.fc.weight.copy_(torch.ones(num_layer_PQC*num_qubits*3, 1))
            self.fc.weight.requires_grad = False
    def forward(self, x):
        x = self.fc(x)
        return x  # this is all angles for unitary gates in PQC
