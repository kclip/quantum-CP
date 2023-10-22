### QCP for quantum data classification
    
-  Main file is 'quantum_classification/main_classification.py'. By running 
    ```
    python main_classification.py --temperature <T for Gibbs state> --mode <'naive' or 'qcp'> --mode_cp <'vanilla' or 'weighted_histo'>
    ```
    one can execute "naive prediction" via 'naive'+'vanilla'; "QCP" via 'qcp'+'weighted_histo'; "CP" via 'qcp'+'vanilla'. Other settings can be found at the beginning of the main file, including the quantum noise drift. 