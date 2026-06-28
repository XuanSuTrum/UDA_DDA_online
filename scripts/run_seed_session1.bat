@echo off
python online_uda_dda.py ^
  --raw_data_dir "F:\Emotion_datasets\preprocess_data\Raw_data\session1" ^
  --output_dir results ^
  --input_type raw ^
  --fs 200 ^
  --target_subjects 1 2 3 ^
  --protocols S0 S1 S2 ^
  --k_list 1 2 3 ^
  --source_epochs 50 ^
  --cal_epochs 10 ^
  --batch_size 64 ^
  --lr 0.001 ^
  --tau 0.8 ^
  --seed 42
