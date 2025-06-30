python main.py --mode extract --raw_data_path "F:/MTC-AIC3/mtcaic3"

python main.py --mode train --data_path "data/features/mi_data.pkl" --save_dir "models"


python main.py \
  --mode infer \
  --model_path models/model_Standard_AdaBoost.pkl \
  --data_path data/features/mi_data.pkl \
  --raw_data_path F:/MTC-AIC3/mtcaic3 \
  --output_path submission.csv \
  --scaler standard
