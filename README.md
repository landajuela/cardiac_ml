<!-- # <img src="./images/logo.png" width="64" valign="middle" alt="Spack"/> Machine Learning for Cardiac Electrical Imaging (cardiac_ml) -->

# Machine Learning for Cardiac Electrical Imaging (cardiac_ml)

The repository contains code for machine learning for cardiac electrical imaging.
The code is specifically designed for the following paper and dataset:
- <em>M. Landajuela, R. Anirudh, J. Loscazo and R. Blake, \
    "Intracardiac Electrical Imaging Using the 12-Lead ECG: A Machine Learning Approach Using Synthetic Data," \
    2022 Computing in Cardiology (CinC), Tampere, Finland, 2022, pp. 1-4, doi: 10.22489/CinC.2022.026.</em> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081783

- <em>M. Landajuela, R, Anirudh, and R. Blake, \
     Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals. \
     In Lawrence Livermore National Laboratory (LLNL) Open Data Initiative. UC San Diego Library Digital Collections. (2022)</em> https://doi.org/10.6075/J0SN094N



The code support two tasks:
- Task 1: Activation Map Reconstruction from ECG
- Task 2: Transmembrane potential Reconstruction from ECG

## Overview

### Installation
To install the code, you can use the following commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Code Structure

The main code is in the following files:
| File | Description |
| --- | --- |
| [`trainable_model.py`](./cardiac_ml/trainable_model.py)| Base class for all the models |
| [`io_util.py`](./cardiac_ml/io_util.py) | Utility functions for input/output |
| [`ml_util.py`](./cardiac_ml/ml_util.py) | Utility functions for machine learning |
| [`data_interface.py`](./cardiac_ml/data_interface.py) | Interface to the data |
| [`learn_ecg2time.py`](./learn_ecg2time.py) | Main code for task 1 |
| [`learn_ecg2vm.py`](./learn_ecg2vm.py) | Main code for task 2 |

### Configuration files

Configuration files can be found in `config/ecg2*.config`. 
The configuration files contain the following sections: 

| Section | Description |
| --- | --- |
| `[DATA]` | Data related parameters |
| `[MODEL]` | Model related parameters |
| `[TRAIN]` | Training related parameters |
| `[SAVE]` | Saving related parameters |

## Data
The data associated with this code is [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106).

<details><summary>Steps to download the dataset (Click to expand)</summary>

To download the data, you can use the following command:
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

</details>

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
## Post-process
``` bash
python3 ./tools/read_training_stats.py trainingStats_errors.h5 -plot True
```

## License

`cardiac_ml` is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE](./LICENSE),
and
[NOTICE](./NOTICE) for details.

LLNL-CODE-850741


