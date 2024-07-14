dataset_name="${1:-videomatte_1920x1080}"
python evaluation/evaluate_hr.py \
--pred-dir /home/xcai/data/pred/$dataset_name \
--true-dir /home/xcai/data/evaluation/$dataset_name