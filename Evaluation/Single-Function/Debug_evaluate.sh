repo_name=$1
model=$2
python debug_code_test.py --repo_name $repo_name --model $model
python debug_code_evaluate.py --repo_name $repo_name --model $model --regenerate