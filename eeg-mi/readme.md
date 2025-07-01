## Scripts to run the code

path-to-folder is the mtc-aic3 main folder

Feature Extraction
```bash
python main.py --mode extract --raw_data_path "path-to-data"
```
Train on validation data
```bash 
python main.py --mode train --data_path "data/features/mi_data.pkl" --save_dir "models"
```
LOSO-Validation
```bash
python main.py --mode loso --model_path models/Model_MinMax_Adaboost.pkl --data_path "data/features/mi_data.pkl"
```
Inference
```bash
python main.py --mode infer --model_path models/model_Standard_AdaBoost.pkl --data_path data/features/mi_data.pkl --raw_data_path "Path-to-data" --output_path submission.csv --scaler standard
```