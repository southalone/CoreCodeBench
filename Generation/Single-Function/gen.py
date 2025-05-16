import subprocess
import re
import ast
import textwrap  # 用于移除额外的缩进
import json
import os
import argparse
# from variable_tracker import extract_lvalues_and_rvalues, extract_lvalues_new
import utils
import argparse
import traceback
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='', help='Repository name')
parser.add_argument('--regenerate', action='store_true', help='Regenerate the test code')
parser.add_argument('--model', type=str, default='gpt4o', help='Model name')
parser.add_argument('--validate_model', type=str, default='qwen-plus-latest', help='Model name')
args = parser.parse_args()

repo_name = args.repo_name
regenerate = args.regenerate
repo_args = utils.get_repo_args(args.repo_name)
mapping_path = repo_args["test_mapping_path"]
find_path = repo_args["find_path"]
root_path = repo_args["root_path"]
output_dir = root_path

directory, _ = os.path.split(mapping_path)
progress_path = os.path.join(directory, f"gen_progress.txt")

print(f"generating repo {repo_name}, \nmapping_path {mapping_path} \nprogress_path {progress_path}\n")
print("-" * 40)
print("-" * 40)

with open(mapping_path, 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(tqdm(file, desc="Processing lines")):
        data = json.loads(line.strip())
        test_file = data.get("test_file", "")
        origin_file = data.get("origin_file", "")
        test_path = test_file
        file_name = origin_file.split('/')[-1].replace('.py', '')
        
        src_transformers_index = origin_file.find(find_path)
        file_path = origin_file[src_transformers_index + len(find_path):origin_file.rfind('/')]
        test_case_dir = os.path.join(output_dir, 'testcases', repo_name, file_path, file_name) 
        print(os.path.join(test_case_dir, 'testcases_info.json'))

        testcases_info_path = os.path.join(test_case_dir, 'testcases_info.jsonl')
        if os.path.exists(testcases_info_path):
            print(f'{file_path} has generated!')
            continue
        

        # 打印或使用提取的信息
        print(f"Test Path: {test_path}")
        print(f"Origin File: {origin_file}")
        print(f"File Name: {file_name}")
        print(f"File Path: {file_path}\n")

        repo_log_dir = os.path.join(output_dir, 'gen_log', repo_name)
        if not os.path.exists(repo_log_dir):
            os.makedirs(repo_log_dir)

        with open (os.path.join(repo_log_dir, f"gen_log.txt"), "a") as f:
            from code_gen import gen_comment


            try: 
                gen_result = gen_comment(repo_name, file_path, file_name, test_path, regenerate, args.model, args.validate_model, output_dir)
            except Exception as e:
                log_dir = os.path.join(output_dir, 'gen_log', repo_name, file_path)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                
                command = [
                    'python', 'code_gen.py',
                    '--file_path', file_path,
                    '--file_name', file_name,
                    '--test_path', test_path,
                    '--repo_name', repo_name
                ]
                with open (os.path.join(log_dir, f"{file_name}.txt"), "w") as output_file:
                    traceback.print_exc(file=output_file)
                    output_file.write(f"An exception occurred: {e.__class__.__name__}: {e}\n")
                    output_file.write("command: \n" + ' '.join(command))
                traceback.print_exc()
                f.write(f"Encountered {e.__class__.__name__}: {e} while generating {test_path} in {os.path.join(file_path, file_name)} \n")
            else:
                if gen_result == "path already exists":
                    f.write(f"Exists {test_path} in {os.path.join(file_path, file_name)}\n")
                else:
                    f.write(f"Generated {test_path} in {os.path.join(file_path, file_name)}\n")

        print("-" * 40)

        
