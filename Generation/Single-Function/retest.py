import subprocess
import re
import ast
import textwrap 
import json
import os
import argparse
from variable_tracker import extract_lvalues_and_rvalues, extract_lvalues_new
import utils
import shutil
import argparse
import traceback
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='', help='Repository name') # transformers, langchain

repo_args = utils.get_repo_args()
root_path = repo_args["root_path"]

parser.add_argument('--if_comments', type=str, default='empty', help='empty or full')
parser.add_argument('--mode', type=str, default='generate', help='generate or evaluate')
parser.add_argument('--model', type=str, default='gpt4o', help='args.model')
parser.add_argument('--output_dir', type=str, default=root_path, help='Output directory for results')

args = parser.parse_args()

repo_name = args.repo_name
output_dir = args.output_dir
repo_args = utils.get_repo_args(args.repo_name)
repo_path = repo_args["repo_path"]
copy_path = repo_args["copy_path"]
mapping_path = os.path.join(args.output_dir, 'testcases', args.repo_name, 'output_testcase_mapping_valid.jsonl')
find_path = repo_args["find_path"]
repo_running_path = repo_args["repo_running_path"]
copy_running_path = repo_args["copy_running_path"]
directory, _ = os.path.split(mapping_path)
progress_path = os.path.join(directory, f"retest_progress.txt")

print(f"testing repo {repo_name}, \nrepo_path {repo_path}, \ncopy_path {copy_path}, \nmapping_path {mapping_path} \nprogress_path {progress_path}\n")

def read_progress():
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return int(f.read().strip())
    return 0

def write_progress(line_num):
    with open(progress_path, 'w') as f:
        f.write(str(line_num))

start_line = read_progress()
# # 临时测试文件
temp_dir = os.path.join(args.output_dir, 'tmp_source_code')
os.makedirs(temp_dir, exist_ok=True)
import tempfile
temp_copy_path = tempfile.mkdtemp(prefix=f'{args.repo_name}_DEBUGEVAL_', dir=temp_dir)
print('COPYING REPO to', temp_copy_path, '.......')
shutil.copytree(src=repo_args['repo_path'], dst=temp_copy_path, dirs_exist_ok=True)
if not temp_copy_path.endswith(os.sep):
    temp_copy_path += os.sep
tmp_running_path = temp_copy_path



with open(mapping_path, 'r', encoding='utf-8') as file:
    for line_num, line in tqdm(enumerate(file), desc="Processing lines", unit="line"):
        # if line_num < start_line:
        #     continue  # 跳过已处理的行

        data = json.loads(line.strip())
        test_file = data.get("test_file", "")
        origin_file = data.get("origin_file", "")
        test_path = test_file
        file_name = origin_file.split('/')[-1].replace('.py', '')

        src_transformers_index = origin_file.find(find_path)
        file_path = origin_file[src_transformers_index + len(find_path):origin_file.rfind('/')]
        test_case_dir = os.path.join(output_dir, 'testcases', repo_name,  file_path, file_name)
        
        # 打印或使用提取的信息
        print(f"Test Path: {test_path}")
        print(f"Origin File: {origin_file}")
        print(f"File Name: {file_name}")
        print(f"File Path: {file_path}")
        
        repo_log_dir = os.path.join(output_dir, 'retest_log', repo_name)
        if not os.path.exists(repo_log_dir):
            os.makedirs(repo_log_dir)
 
        with open (os.path.join(repo_log_dir, f"retest_log.txt"), "a") as f:
            
            from code_retest import retest_code
            try: 
                retest_code(args.if_comments, args.mode, args.model, repo_name, file_path, file_name, test_path, tmp_running_path, output_dir)
            except Exception as e:
                print("An exception occurred:", e)
                log_dir = os.path.join(output_dir, 'retest/log', repo_name, file_path)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                command = [
                    'python', 'code_retest.py',
                    '--if_comments', args.if_comments,
                    '--mode', args.mode,
                    '--model', args.model,
                    '--file_path', file_path,
                    '--file_name', file_name,
                    '--test_path', test_path,
                    '--repo_name', repo_name
                ]
                with open (os.path.join(log_dir, f"{file_name}_retest_py.txt"), "w") as output_file:
                    traceback.print_exc(file=output_file)
                    output_file.write(f"An exception occurred: {e.__class__.__name__}: {e}\n")
                    output_file.write("command: \n" + ' '.join(command))

                f.write(f"Encountered error while retesting {test_path} in {os.path.join(file_path, file_name)} \n")
            else: 
                f.write(f"Retested {test_path} in {os.path.join(file_path, file_name)}\n")
                
        print("-" * 40)
        write_progress(line_num + 1)

shutil.rmtree(temp_copy_path)
print('Removed temp path:', temp_copy_path)
