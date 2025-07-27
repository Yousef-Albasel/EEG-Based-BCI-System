# Deep SSVEP EEG Classification

A deep learning model for classifying SSVEP (Steady-State Visual Evoked Potential) EEG signals using PyTorch.

## To run the inferance script on the test data easliy 

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```
2. run the inferance:
```bash
python main.py --mode infer --data_path "add dataset path here"
```

## Usage

The model provides two main commands: training and inference.

### Training

Train the SSVEP classification model using your dataset:

```bash
python main.py --mode train --data_path /path/to/your/data
```

#### Training Options

- `--data_path`: Path to the directory containing your training data (required)
#### Example with custom parameters:
```bash
python main.py --mode train --data_path "../eeg-data"
```

### Inference

Run inference on new data using a trained model:

```bash
python main.py infer --model-path models/ssvep_model.pt --data_path /path/to/test/data
```

#### Inference Options

- `--data_path`: Path to the data for inference (required)
- `--model_path`: Path to the trained model (default: models/ssvep_model.pt)
- `--encoder_path`: Path to the label encoder (default: models/encoder.joblib)