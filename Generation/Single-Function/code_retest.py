# -*- coding: utf-8 -*-
from server import LocalClient
import json
import os
import shutil
from utils import generate_xlsx, read_log, extract_code
import argparse
import pandas as pd
import utils
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument('--if_comments', type=str, default='empty', help='empty or full')
parser.add_argument('--mode', type=str, default='generate', help='generate or evaluate')
parser.add_argument('--model', type=str, default='llama', help='args.model')

repo_args = utils.get_repo_args()
root_path = repo_args["root_path"]

parser.add_argument('--repo_name', type=str, default='transformers', help='Repository name')
parser.add_argument('--file_path', type=str, default='src/transformers/', help='File path without extension')
parser.add_argument('--file_name', type=str, default='audio_utils', help='File name without extension')
parser.add_argument('--test_path', type=str, default=root_path + 'Source_Copy/transformers/tests/utils/test_audio_utils.py', help='Output directory for results')
parser.add_argument('--output_dir', type=str, default=root_path, help='Output directory for results')

args = parser.parse_args()

from pathlib import Path

def find_diff_segments(path1, path2):
    """找到两个路径中不同的段落"""
    parts1 = Path(path1).parts
    parts2 = Path(path2).parts
    
    # 找到第一个不同的索引
    start = None
    for i, (p1, p2) in enumerate(zip(parts1, parts2)):
        if p1 != p2:
            start = i
            break
    
    if start is None:
        return None, None  # 路径完全相同
    
    # 找到后续第一个相同的目录
    end = None
    for j in range(start, min(len(parts1), len(parts2))):
        if parts1[j] == parts2[j]:
            end = j
            break
    
    return (start, end) if end else (start, None)

def replace_path_segment(original_path, reference_path, replacement_path):
    """动态替换路径段"""
    orig_parts = Path(original_path).parts
    ref_parts = Path(reference_path).parts
    repl_parts = Path(replacement_path).parts
    
    # 找到差异段落
    start, end = find_diff_segments(Path(original_path), Path(reference_path))
    
    if not start:
        return str(original_path)  # 没有需要替换的部分
    
    # 确定替换范围
    replace_range = slice(start, end)
    
    # 构建新路径
    new_parts = (
        repl_parts[:start] +  # 保留替换路径前缀
        repl_parts[start:end] +  # 替换差异部分
        orig_parts[end:]  # 保留原始路径后缀
    )
    
    return str(Path(*new_parts))

def retest_code(if_comments, mode, model, repo_name, file_path, file_name, test_path, tmp_running_path,  output_dir):

    args.if_comments = if_comments
    args.mode = mode
    args.model = model
    args.repo_name = repo_name
    args.file_path = file_path
    args.file_name = file_name
    args.test_path = test_path
    args.output_dir = output_dir


    repo_args = utils.get_repo_args(args.repo_name)
    repo_path = repo_args["repo_path"]
    temp_running_path = os.path.join(tmp_running_path, repo_args["relative_running_path"])
    test_tmp_path = os.path.join(tmp_running_path, repo_args["relative_test_path"])
    test_path = replace_path_segment(original_path=test_path, reference_path=test_tmp_path, replacement_path=test_tmp_path)

    test_case_dir = os.path.join(output_dir, 'testcases', repo_name,  file_path, file_name)

    print(f"testing repo {repo_name}, repo_path {repo_path}, temp_path {tmp_running_path}")
    print(f"testing file {file_name} in {file_path}.")
    print(f"test path: {test_path}")

    source_code_path = os.path.join(repo_path, file_path, f'{file_name}.py')

    result_dir = os.path.join(output_dir, 'retest', repo_name, file_path,  file_name)
    save_name = "retest"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 修改Sourse_Copy中的文件
    testcases = {}
    with open(os.path.join(test_case_dir, 'testcases_info.jsonl'),'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            testcases[data['func']] = (data['id'], data['code'])


    if os.path.exists(os.path.join(test_case_dir, f'testcases_valid_{save_name}_info.jsonl')):
        os.remove(os.path.join(test_case_dir, f'testcases_valid_{save_name}_info.jsonl'))


    with open(source_code_path, 'r') as f:
        source_code = f.read().splitlines()

    with open(os.path.join(test_case_dir, f'testcases_valid_{save_name}_info.jsonl'), 'w', encoding='utf-8') as file:
        for func, item in testcases.items():
            test_id, testcase = item
            shutil.copy(source_code_path, source_code_path.replace(repo_path, tmp_running_path))
            print('FUNC',func)
            file_exists = os.path.exists(os.path.join(result_dir,f'''{func}_completed_{save_name}.py'''))
            if not file_exists or args.mode == 'generate':
                completed_key_block = "_ = 1"
                
                # 对齐缩进
                prefix = source_code[:testcase['key_block_start_lineno'] -1]
                suffix = source_code[testcase['key_block_end_lineno']:]
                indented_completed_key_block = utils.align_indent(completed_key_block.splitlines(), source_code[testcase['key_block_start_lineno']-1:testcase['key_block_end_lineno']], prefix, suffix )
                compeletd_code = prefix + indented_completed_key_block + suffix

                with open(os.path.join(result_dir,f'''{func}_completed_{save_name}.py'''),'w') as f:
                    f.write('#test_copy\n' + '\n'.join(compeletd_code))
                
                shutil.copy(os.path.join(result_dir,f'''{func}_completed_{save_name}.py'''), source_code_path.replace(repo_path, tmp_running_path))
        
                os.chdir(tmp_running_path)
                
                BASH = f'''http_proxy=http://10.217.142.137:8080  https_proxy=http://10.217.142.137:8080 PYTHONPATH={temp_running_path} timeout 60 pytest {test_path} --tb=long > {result_dir}/{func}_{save_name}.log'''    
                print(BASH)
                status = os.system(BASH)
                failed = 0
                passed = 0
                if status == 0:
                    passed, skipped, failed = read_log(os.path.join(result_dir, f'''{func}_{save_name}.log'''))
                if (status != 0 and status != 1024) or failed != 0:
                    with open(os.path.join(test_case_dir, 'testcases_info.jsonl'),'r') as f:
                        for line in f.readlines():
                            data = json.loads(line)
                            if data['func'] == func:
                                data["unitest"] = passed
                                file.write(json.dumps(data, ensure_ascii=False) + '\n')

            shutil.copy(source_code_path, source_code_path.replace(repo_path, tmp_running_path)) # 复原文件

    

if __name__ == "__main__":
    retest_code(args.if_comments, args.mode, args.model, args.repo_name, args.file_path, args.file_name, args.test_path, args.output_dir)
    