# Intracardiac electrical imaging using the 12-lead ECG: a machine learning approach using synthetic data

## Links
- Paper : https://cinc.org/2022/Program/accepted/26_Preprint.pdf
- Dataset : https://library.ucsd.edu/dc/object/bb29449106

## Install
- `torch`
- `tqdm`
- `h5py`

## Code
```
cardiac_ml
│___ trainable_model.py
│___ io_util.py
│___ ml_util.py
│___ data_interface.py
```

## Config
Configuration files in `config/ecg2*.config`.
Download data from https://library.ucsd.edu/dc/object/bb29449106 and point to it here:
```
[DATA]
datapaths_train = POINTER TO https://library.ucsd.edu/dc/object/bb29449106
datapaths_val = POINTER TO https://library.ucsd.edu/dc/object/bb29449106
```

## Run
```
python3 learn_ecg2time.py config/hyperparams_ecg2time.config
python3 learn_ecg2vm.py config/hyperparams_ecg2vm.config
```
Note: You can run this form `ecg2*_input_deck`:
```
python3 ../learn_ecg2time.py ../config/hyperparams_ecg2time.config
python3 ../learn_ecg2vm.py ../config/hyperparams_ecg2vm.config
```
Postprocess:
```
python3 ../tools/read_training_stats.py trainingStats_errors.h5 -plot True
```