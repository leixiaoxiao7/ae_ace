# ACE Brain Data Classification Experiments

This repository contains research code for preparing GMWM-style brain features, training neural-network classifiers, and experimenting with a reinforcement-learning based classifier. The current workflows focus on binary cohort labels stored in the processed dataset as `ace`.

## What is in here

- Brain-feature preprocessing from multi-index Excel files into 3D numpy arrays.
- Several TensorFlow/Keras classifier variants, including CNN, ResNet-style CNN, CNN-LSTM, and attention-based models.
- A DQN-based classification workflow built on `gym` and `keras-rl2`.
- Utility scripts for validating the data, loading `.mat` files, and plotting training curves.
- Saved model weights, checkpoints, and training history plots under `models/`.

## Repository layout

- `main.py` - trains the `CNN4` model on processed 3D GMWM data.
- `main_gmwmProc.py` - trains the `CNN6` model on the same processed dataset.
- `multi_nns.py` - compares multiple deep-learning classifiers on the processed brain features.
- `rlClassify.py` - reinforcement-learning classification pipeline using a custom Gym environment.
- `main_mat.py` - helper for loading legacy MATLAB data.
- `verify_data.py` - checks that the Excel, CSV, MAT, and pickle sources can be loaded correctly.
- `funcs/` - shared preprocessing, feature selection, plotting, environment, and model-definition helpers.
- `autoencoders/` - standalone MNIST autoencoder experiments.
- `data/` - processed pickles plus raw and intermediate data files.
- `models/` - saved `.h5` weights, checkpoints, and generated plots.

## Data files

The main preprocessing code expects these source files to be available in the project root:

- `subjBrainDataNarrGen_360plain.xlsx`
- `subjBrainDataConnGen_360plain.xlsx`
- `XXL_ACEcohort395.xlsx`

Processed features are stored in `data/gmwm3D.pkl`. That pickle contains:

- `gmwm3D`
- `gmwm3D_extend`
- `df`
- `gmwmDimensions`
- `uniqueHeaders`
- `otherFeatureHeaders`
- `siteList`

If you need to regenerate the pickle, the preprocessing logic lives in `funcs/dataCook.py` in `cookGMWM_withDemo(...)`.

## Environment

This repo has been run in two different Python/TensorFlow setups:

- Python 3.9.7 with TensorFlow/Keras 2.9 and `keras-rl2`
- Python 3.12.11 with TensorFlow 2.16.2 and Keras 2.11.3

Common packages used across the repo:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `openpyxl`

Optional or RL-specific packages:

- `gym`
- `keras-rl2`
- `pymysql`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib seaborn tensorflow openpyxl
pip install gym keras-rl2 pymysql
```

Then verify the bundled data:

```bash
python verify_data.py
```

## Running the main scripts

Run everything from the repository root so the relative paths in the scripts resolve correctly.

```bash
python main.py
python main_gmwmProc.py
python multi_nns.py
python rlClassify.py
```

Notes:

- `main.py` and `main_gmwmProc.py` load `data/gmwm3D.pkl`, train a binary classifier, and save the best checkpoint plus a training-history plot.
- `multi_nns.py` trains several alternative architectures and saves a comparison plot.
- `rlClassify.py` uses a custom `ClassifyEnv` in `funcs/ICMDP_Env_xxl_multiClass.py` and writes RL outputs under `models/rlSave/`.
- `verify_data.py` is a good first command if you want to confirm that the data files are in the right place.

## Typical outputs

- Best model weights such as `CNN4_best.h5`, `CNN6_best.h5`, or similar checkpoint files.
- Training-history images such as `CNN4_training_history_*.png`.
- Confusion matrices and classification reports printed to the console.
- RL summaries saved as `.png` and `.txt` files under `models/rlSave/`.

## Notes

- Many scripts are research prototypes and assume binary labels from `gmwmDF["ace"]`.
- Several modules use `sys.path.append('./funcs')`, so running from the repo root matters.
- The codebase mixes older and newer TensorFlow/Keras conventions, so matching the intended environment is important if you hit import or compatibility errors.
