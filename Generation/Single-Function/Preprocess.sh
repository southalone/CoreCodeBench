model=$1

python repo_test_file_mapper.py --repo_name $repo_name 
python test_all_test.py --repo_name $repo_name
python functionTree_generate.py --repo_name $repo_name