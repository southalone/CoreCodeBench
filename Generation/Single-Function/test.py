import subprocess
import re
import ast
import textwrap  # 用于移除额外的缩进
import json
import os
import argparse
from variable_tracker import extract_lvalues_and_rvalues, extract_lvalues_new
import utils
import argparse
import traceback
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--if_comments', type=str, default='full', help='empty or full')
parser.add_argument('--mode', type=str, default='generate', help='generate or evaluate')#transformers,langchain,datachain
parser.add_argument('--model', type=str, default='o1mini', help='args.model')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true')
parser.add_argument('--language',type=str, default='ch', help='ch/en')
parser.add_argument('--repo_name', type=str, default='langchain', help='Repository name') # transformers, langchain

args = parser.parse_args()

repo_name = args.repo_name
output_dir = args.output_dir
repo_args = utils.get_repo_args(args.repo_name)
repo_path = repo_args["repo_path"]
copy_path = repo_args["copy_path"]
mapping_path = repo_args["test_mapping_path"]
find_path = repo_args["find_path"]

directory, _ = os.path.split(mapping_path)
progress_path = os.path.join(directory, f"test_progress_{args.if_comments}_{args.mode}_{args.model}.txt")

print(f"testing repo {repo_name}, \nrepo_path {repo_path}, \ncopy_path {copy_path}, \nmapping_path {mapping_path} \nprogress_path {progress_path}\n")

# 创建临时文件
if args.mode == 'evaluate':
    parent_dir = os.path.dirname(repo_args['repo_path'].rstrip('/\\'))
    tmp_copy_base = os.path.join(parent_dir, 'tmp')
    os.makedirs(tmp_copy_base, exist_ok=True)
    import tempfile
    import shutil
    temp_copy_path = tempfile.mkdtemp(prefix=f"{repo_name}_{args.model}_", dir=tmp_copy_base)
    print(repo_args['repo_path'])
    print(temp_copy_path)
    shutil.copytree(repo_args['repo_path'], temp_copy_path, dirs_exist_ok=True)
    if not temp_copy_path.endswith(os.sep):
        temp_copy_path += os.sep
else:
    temp_copy_path = repo_args['copy_running_path']


with open(mapping_path, 'r', encoding='utf-8') as file:
    for line_num, line in tqdm(enumerate(file), desc="Processing lines", unit="line"):
        data = json.loads(line.strip())
        test_file = data.get("test_file", "")
        origin_file = data.get("origin_file", "")
        #test_path = test_file
        file_name = origin_file.split('/')[-1].replace('.py', '')

        src_transformers_index = origin_file.find(find_path)
        file_path = origin_file[src_transformers_index + len(find_path):origin_file.rfind('/')]
        test_file_relative = os.path.relpath(test_file, copy_path)
        test_path = os.path.join(temp_copy_path, test_file_relative)
        print(test_path)
        #test_file_path = os.path.join(tmp_repo_path, test_path)
        # 打印或使用提取的信息
        print(f"Test relative path: {test_file_relative}")
        print(f"Test Path: {test_path}")
        print(f"Origin File: {origin_file}")
        print(f"File Name: {file_name}")
        print(f"File Path: {file_path}")
        
        repo_log_dir = os.path.join(output_dir, 'test_log', repo_name)
        if not os.path.exists(repo_log_dir):
            os.makedirs(repo_log_dir)

        with open (os.path.join(repo_log_dir, f"test_log_{args.mode}_{args.if_comments}_{args.model}.txt"), "a") as f:

            try: 
                if args.language == 'ch':
                    from code_test import test_code
                    #print(args.if_comments, args.mode, args.model, repo_name, file_path, file_name, test_path, temp_copy_path, args.regenerate, output_dir)
                    test_code(args.if_comments, args.mode, args.model, repo_name, file_path, file_name, test_path, temp_copy_path, args.regenerate, output_dir)
                elif args.language == 'en':
                    from code_test_en import test_code
                    test_code(args.if_comments, args.mode, args.model, repo_name, file_path, file_name, test_path, temp_copy_path, args.regenerate, output_dir)
                    
            except Exception as e:
                log_dir = os.path.join(output_dir, 'test_log', repo_name, args.if_comments, args.mode, args.model, file_path)
                print('Error occurred:', e)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                command = [
                    'python3', 'code_test.py',
                    '--if_comments', args.if_comments,
                    '--mode', args.mode,
                    '--model', args.model,
                    '--file_path', file_path,
                    '--file_name', file_name,
                    '--test_path', test_path,
                    '--repo_name', repo_name,
                    '--temp_copy_path', temp_copy_path,
                ]
                if args.regenerate:
                    command += '--regenerate'
                with open (os.path.join(log_dir, f"{file_name}.txt"), "w") as output_file:
                    traceback.print_exc(file=output_file)
                    output_file.write(f"An exception occurred: {e.__class__.__name__}: {e}\n")
                    output_file.write("command: \n" + ' '.join(command))

                f.write(f"Encountered error while testing {test_path} in {os.path.join(file_path, file_name)} \n")
            else: 
                f.write(f"Tested {test_path} in {os.path.join(file_path, file_name)}\n")
                # env = os.environ.copy()
                # command = [
                #     'python', 'code_dev_test_auto.py',
                #     '--if_comments', args.if_comments,
                #     '--mode', args.mode,
                #     '--model', args.model,
                #     '--file_path', file_path,
                #     '--file_name', file_name,
                #     '--test_path', test_path,
                #     '--repo_name', repo_name
                # ]
                # print("Running command: " + ' '.join(command))

                # result = subprocess.run(
                #     command,
                #     stdout=output_file,
                #     stderr=output_file,
                #     text=True
                # )
                # if not result.returncode == 0:
                #     f.write(f"{test_path} encountered error\n")
                # else:
                #     f.write(f"test_path: {test_path}, file_path: {file_path+ "/" + file_name} has been generated\n")
        print("-" * 40)
        #write_progress(line_num + 1)

if args.mode == 'evaluate' and temp_copy_path:
    shutil.rmtree(temp_copy_path)
  