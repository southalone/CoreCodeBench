repo_name=$1
model=$2
python tdd_evaluate.py --repo_name $repo_name --model $model
python tdd_evaluate_run.py --repo_name $repo_name --model $model --regenerate