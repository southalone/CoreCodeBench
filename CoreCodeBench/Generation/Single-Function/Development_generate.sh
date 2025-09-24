model=$1

python gen.py --repo_name $repo_name
python retest.py --repo_name $repo_name
