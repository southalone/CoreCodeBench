# -*- coding: utf-8 -*-
import json
import os
import sys
import shutil
# 获取当前脚本所在目录的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
from utils import generate_xlsx, read_log, extract_code
import argparse
import pandas as pd
import utils
import time
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DeepSeekR1', help='args.model')
parser.add_argument('--gen_model', type=str, default='mix', help='Repository name')
parser.add_argument('--rewrite_model',type=str, default='mix', help='rewrite model')
parser.add_argument('--repo_name', type=str, default='open-iris', help='Repository name')
parser.add_argument('--test_path', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/transformers/tests/utils/test_audio_utils.py', help='Output directory for results')
parser.add_argument('--output_dir', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='regenerate/reevaluate everything')
parser.add_argument('--mode',type=str, default='empty', choices=['empty','log'],help='provide log in the content')
parser.add_argument('--type', type=str, default='bugfix', help='dev, TDD, bugfix')

args = parser.parse_args()
repo_args = utils.get_repo_args(args.repo_name)
args.debug_output_dir = os.path.join(args.output_dir, 'experiments', args.type)
args.debug_testcases_dir = os.path.join(args.output_dir, 'DEBUG', 'testcases_logic')

# print(utils.get_response('请写一段快速排序的代码', model=args.model))

def complete_code_with_log(new_code, log, test_code, model):
    chat_message = f'''In the following code snippet, there is a buggy code section between `<buggy code begin>` and `<buggy code end>`. I've provided the corresponding unit test file and pytest error messages. Please analyze the given context and rewrite the erroneous code segment.
Please format the rewritten function block in markdown (```python```), including only the rewritten content between `<buggy code begin>` and `<buggy code end>`, without including the `<buggy code begin>` and `<buggy code end>` tags.
**Note**: Please ensure that your completed code block maintains the indentation of the original code context.

Code snippet:
```python
{new_code}
```
Unit test code:
```python
{test_code}
```
Test error log：
```
{log}
```
'''
    res = utils.get_response(chat_message, model)
    return extract_code(res), res

def complete_code(new_code, model):
    chat_message = f'''
In the following code snippet, there is a buggy code section between `<buggy code begin>` and `<buggy code end>`.  Please analyze the given context and rewrite the erroneous code segment.
Please format the rewritten function block in markdown (```python```), including only the rewritten content between `<buggy code begin>` and `<buggy code end>`, without including the `<buggy code begin>` and `<buggy code end>` tags.
**Note**: Please ensure that your completed code block maintains the indentation of the original code context.
Code snippet:
```python
{new_code}
```
'''
    res = utils.get_response(chat_message, model)
    return extract_code(res), res

def test_gen_code(id, model, repo_name, file_path, file_name, test_path_list, prob_info, output_dir = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/"):  
    args.model = model
    args.repo_name = repo_name
    args.file_path = file_path
    args.file_name = file_name
    args.output_dir = output_dir

    repo_args = utils.get_repo_args(args.repo_name)
    repo_path, copy_path = repo_args["repo_path"], repo_args["copy_path"]
    


    debug_result_dir = args.debug_output_dir


    source_code_path = os.path.join(repo_args['repo_running_path'], file_path, f'{file_name}.py')
    result_dir = os.path.join(debug_result_dir, repo_name, args.model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if os.path.exists(os.path.join(result_dir, f'{id}_completed_code.py')):
        return
    if os.path.exists(os.path.join(result_dir, f'{id}_buggy_code.py')) and not args.regenerate:
        return 

    with open(source_code_path, 'r') as f:
        source_code = f.read().splitlines()
    
    prefix = '\n'.join(source_code[:prob_info['func_start_lineno']-1])
    suffix = '\n'.join(source_code[prob_info['func_end_lineno']: ])
    new_code = prefix +  prob_info['new_func_code'] +  suffix
    if args.mode == 'log':
        log_path = os.path.join(args.debug_testcases_dir, args.repo_name, f"{args.gen_model}-{args.rewrite_model}", 'retest_{}.log'.format(id))
        with open(log_path, 'r') as f:
            log_file = f.read()
        test_file_path = [ os.path.join(repo_args['repo_path'], test_path) for test_path in test_path_list]
        test_code = ''
        for test_path in test_file_path:
            with open(test_path, 'r') as f:
                test_code += f.read() + '\n'
       
        try:
            completed_code, response = complete_code_with_log(new_code, log_file, test_code, args.model)
        except Exception as e:
            # if exceed max length of prompt
            print('Error:', e)
            completed_code, response = complete_code(new_code, args.model)

    elif  args.mode == 'empty':
        completed_code, response = complete_code(new_code, args.model)
    # print(id, prob_info)
    prefix = source_code[:prob_info['key_block_start_lineno'] -1]
    suffix = source_code[prob_info['key_block_end_lineno']:]
    completed_code = utils.remove_common_prefix(utils.remove_common_indent(completed_code),utils.remove_common_indent('\n'.join(str(x) for x in source_code[prob_info['func_start_lineno']-1:prob_info['key_block_start_lineno']-1])))
    indented_completed_key_block = utils.align_indent(completed_code.splitlines(), source_code[prob_info['key_block_start_lineno']-1:prob_info['key_block_end_lineno']], prefix, suffix )
    completed_code = prefix + indented_completed_key_block + suffix

    with open(os.path.join(result_dir, f'{id}_completed_code.py'),'w') as f:
        f.write('# debug code\n' + '\n'.join(completed_code))
    with open(os.path.join(result_dir, f'{id}_buggy_code.py'),'w') as f:
        f.write('# buggy code\n' + new_code)
    with open(os.path.join(result_dir, f'{id}_source_code.py'),'w') as f:
        f.write('\n'.join(source_code)) 
    with open(os.path.join(result_dir, f'{id}_response.py'),'w') as f:
        f.write(response)

 
if __name__ == "__main__":
    testcases = []
    with open(os.path.join(args.debug_testcases_dir,  args.repo_name, f"{args.gen_model}-{args.rewrite_model}", f'buggy_{args.repo_name}_new.jsonl'),'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            testcases.append(data)
    for testcase in tqdm(testcases):
        id = testcase['id']
        file_path, filename_with_extension = os.path.split(testcase['origin_file'])
        file_name, _ = os.path.splitext(filename_with_extension)
        test_path_list = testcase['test_list']
        prob_info = testcase['prob_info']
        test_gen_code(id, args.model, args.repo_name, file_path, file_name, test_path_list, prob_info, args.output_dir)
    