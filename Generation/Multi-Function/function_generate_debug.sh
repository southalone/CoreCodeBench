repo_name=$1

python function_generate_debug.py --repo_name $repo_name
python function_debug_test_all.py --repo_name $repo_name --model retest --regenerate --if_comments debug
python function_debug_test_all.py --repo_name $repo_name --model retest --mode evaluate --regenerate --if_comments debug
python function_combine.py --repo_name $repo_name --if_comments debug