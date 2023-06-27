<!-- # <img src="./images/logo.png" width="64" valign="middle" alt="Spack"/> Machine Learning for Cardiac Electrical Imaging (cardiac_ml) -->

# Machine Learning for Cardiac Electrical Imaging (cardiac_ml)

PyTorch code for cardiac electrical imaging using the 12-lead ECG. This code is associated with the paper from CINC 2022. If you use this code, please cite:
- Paper : https://cinc.org/2022/Program/accepted/26_Preprint.pdf
- Dataset : https://library.ucsd.edu/dc/object/bb29449106

The code support two tasks:
- Task 1: Activation Map Reconstruction from ECG
- Task 2: Transmembrane potential Reconstruction from ECG

## Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Code Structure
The code is organized as follows:
| File | Description |
| --- | --- |
| `cardiac_ml/trainable_model.py` | Base class for all the models |
| `cardiac_ml/io_util.py` | Utility functions for input/output |
| `cardiac_ml/ml_util.py` | Utility functions for machine learning |
| `cardiac_ml/data_interface.py` | Interface to the data |
| `learn_ecg2time.py` | Main code for task 1 |
| `learn_ecg2vm.py` | Main code for task 2 |

## Configuration files

Configuration files can be found in `config/ecg2*.config`. 
The configuration files contain the following sections: *[DATA], [MODEL], [TRAIN], [SAVE]*.

### Data
The data associated with this code is [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106).

**Note**:To download the data, you can use the following command:
```bash
source download_intracardiac_dataset.sh
```

Once, it is downloaded, you can point to the data using the `datapaths_train` and `datapaths_val` in the configuration file.
For example, the configuration file for `ecg2time` looks like:
```txt
[DATA]
datapaths_train = [full path of intracardiac_dataset]/data_hearts_dd_0p2
datapaths_val = [full path of intracardiac_dataset]/data_hearts_dd_0p2
```
**Note**: You might want to change the train and validation data path to point to your split of the data.

## Run
```bash
python3 learn_ecg2time.py config/ecg2time.config
python3 learn_ecg2vm.py config/ecg2vm.config
```
**Note**: You can create an input deck directory for the training `ecg2*_input_deck` 
and run the code from the input deck directory.
```bash
python3 ../learn_ecg2time.py ../config/ecg2time.config
python3 ../learn_ecg2vm.py ../config/ecg2vm.config
```
## Postprocess
```
python3 ./tools/read_training_stats.py trainingStats_errors.h5 -plot True
```

## License

`cardiac_ml` is distributed under the terms of the MIT license.
All new contributions must be made under the MIT license.

See [LICENSE](./LICENSE),
and
[NOTICE](./NOTICE) for details.

LLNL-CODE-850741


