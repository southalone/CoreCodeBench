repo_name=$1
model=$2

python test.py --repo_name $repo_name --model $model --if_comments full --mode generate
python test.py --repo_name $repo_name --model $model --if_comments empty --mode generate
python test.py --repo_name $repo_name --model $model --if_comments full --mode evaluate
python test.py --repo_name $repo_name --model $model --if_comments empty --mode evaluate
python IG_filter.py --repo_name $repo_name