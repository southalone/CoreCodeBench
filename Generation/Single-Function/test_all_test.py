import subprocess
import re
import ast
import textwrap  # 用于移除额外的缩进
import json
import os
import argparse
# from variable_tracker import extract_lvalues_and_rvalues, extract_lvalues_new
from utils import read_log
import utils
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='', help='name of repo')
args = parser.parse_args()
repo_name = args.repo_name

repo_args = utils.get_repo_args(repo_name)
root_path = repo_args["root_path"]

mapping_path = root_path + f"testcases/{repo_name}/testcase_mapping.jsonl"
valid_mapping_path = root_path + f"testcases/{repo_name}/output_testcase_mapping_valid.jsonl"
invalid_mapping_path = root_path + f"testcases/{repo_name}/output_testcase_mapping_invalid.jsonl"
progress_path = root_path + f"testcases/{repo_name}/progress.txt"
output_dir = root_path
repo_info_path = root_path + 'repo_info.json'

with open(repo_info_path, 'r') as file:
    repo_info = json.load(file)

if repo_name in repo_info:
    repo_data = repo_info[repo_name]
    copy_path = repo_data.get('copy_path', '')
    repo_path = repo_data.get('repo_path', '')
    running_path_relative = repo_data.get('_running_path', '').lstrip('/')
    copy_running_path = os.path.join(copy_path, running_path_relative)
else:
    print(f"Repository '{repo_name}' not found in the JSON file.")

def read_progress():
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return int(f.read().strip())
    return 0

def write_progress(line_num):
    with open(progress_path, 'w') as f:
        f.write(str(line_num))
    # pass
results_df = pd.DataFrame(columns=['test_id', 'passed', 'skipped', 'failed'])

start_line = read_progress()


with open(mapping_path, 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(tqdm(file, desc="Processing lines")):
        if line_num < start_line:
            continue  # 跳过已处理的行

        data = json.loads(line.strip())
        test_file = data.get("test_file", "")
        origin_file = data.get("origin_file", "")
        test_path = test_file
        file_name = origin_file.split('/')[-1].replace('.py', '')

        src_transformers_index = origin_file.find('Source_Copy/transformers/')
        file_path = origin_file[src_transformers_index + len("Source_Copy/transformers/"):origin_file.rfind('/')]

        # 打印或使用提取的信息
        print(f"Test Path: {test_path}")
        print(f"Origin File: {origin_file}")
        print(f"File Name: {file_name}")
        print(f"File Path: {file_path}")
        print("-" * 40)
        result_dir = os.path.join(output_dir, 'general_test', repo_name, file_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # 执行命令
        os.chdir(repo_path)
        # BASH = f'''PYTHONPATH={copy_path}/src pytest {test_path} --tb=long > {result_dir}/{file_name}_origin_test_result.log'''
        # os.system(BASH)
        # passed, skipped, failed = read_log(os.path.join(result_dir, f'''{file_name}_origin_test_result.log'''))

        BASH = f'''http_proxy=http://10.217.142.137:8080  https_proxy=http://10.217.142.137:8080 PYTHONPATH={copy_running_path} timeout 120 pytest {test_path} --tb=long > {result_dir}/{file_name}_origin_test_result.log'''
        print(BASH)
        exit_code = os.system(BASH)

        # env = os.environ.copy()
        # env["PYTHONPATH"] = repo_running_path
        # env["http_proxy"] = "http://10.229.18.30:8412"
        # env["https_proxy"] = "http://10.229.18.30:8412"
        # pytest_command = f"pytest {test_path} --tb=long"
        # result = subprocess.run(pytest_command, env=env, timeout=120, shell=True)

        # print(result)

        # 检查命令的退出状态
        if exit_code == 0:
            passed, skipped, failed = read_log(os.path.join(result_dir, f'''{file_name}_origin_test_result.log'''))
        else:
            print(exit_code)
            print(f"Test {test_path} exceeded time limit and was terminated.")
            passed, skipped, failed = 0, 0, 1  # 设置 failed 为 0
        results_df = results_df._append({'test_id': [test_file], 'passed': [passed], 'skipped': [skipped], 'failed': [failed]}, ignore_index=True)
        if failed == 0 and (passed > 0.5*(passed + skipped + failed)):
            valid_data = {
                "test_file": test_file,
                "origin_file": origin_file,
                "pytest": {
                    "passed": passed,
                    "skipped": skipped
                }
            }
            
            with open(valid_mapping_path, 'a') as valid_file:
                valid_file.write(json.dumps(valid_data) + "\n")
                # valid_file.write(line)
            print(f"{test_file} is valid")
        else:
            with open(invalid_mapping_path, 'a') as valid_file:
                valid_file.write(f"{test_file} failed{failed} passed{passed} skipped{skipped}\n")
            print(f"{test_file} failed{failed} passed{passed} skipped{skipped}")
        
        write_progress(line_num + 1)


        