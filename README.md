## Quantum Conformal Prediction (QCP)

This repository contains code for "[Quantum Conformal Prediction for Reliable Uncertainty Quantification in Quantum Machine Learning](https://arxiv.org/abs/2304.03398)" - 
Sangwoo Park and Osvaldo Simeone.

### Dependencies

This program is written in python 3.9.7 and uses PyTorch 1.10.2.

### Basic Usage

- All the essential functionalities of QCP can be found in the class file 'set_predictors/quantum_conformal_prediction.py'.
- PQC with different angle encodings (fixed, linear, non-linear angle encoding, see Fig. 9) can be found in the class file 'quantum_circuit/PQC.py'.
- In order to deploy the above PQC to IBM Quantum NISQ devices, 'quantum_circuit/PQC_with_qiskit.py' can be useful.

### Unsupervised Learning (Density Learning)
    
-  Main file is 'main_density_learning.py', while the 'runs/density_learning' folder contains the required running shell scripts. 


### Supervised Learning (Regression)
    
-  Main file is 'main_regression.py', while the 'runs/regression' folder contains the required running shell scripts. 
