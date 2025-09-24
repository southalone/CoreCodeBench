repo_name=$1

python function_generate_tdd.py --repo_name $repo_name
python function_tdd_test_all.py --repo_name $repo_name --model retest
python function_tdd_test_all.py --repo_name $repo_name --model retest --mode evaluate --regenerate
python function_combine.py --repo_name $repo_name --if_comments tdd