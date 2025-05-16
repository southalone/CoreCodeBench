repo_name=$1
model=$2
python evaluate.py --repo_name $repo_name --model $model
python evaluate_run.py --repo_name $repo_name --model $model