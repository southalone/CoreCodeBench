#!/bin/bash
repo_name=$1

python function_generate_difficult.py --repo_name $repo_name --generate_tools
python function_test_all.py --repo_name $repo_name --model retest --if_comments sub
python function_test_all.py --repo_name $repo_name --model retest --mode evaluate --if_comments sub
python function_combine.py --repo_name $repo_name --if_comments sub