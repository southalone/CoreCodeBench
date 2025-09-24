# -*- coding: utf-8 -*-
import json
import os
import sys
import shutil
# 获取当前脚本所在目录的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
from utils import generate_xlsx, read_log, extract_code_loose
import argparse
import pandas as pd
import utils
import time
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DeepSeekR1', help='args.model')

parser.add_argument('--repo_name', type=str, default='open-iris', help='Repository name')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='regenerate/reevaluate everything')
parser.add_argument('--type', type=str, default='TDD', help='dev, TDD, bugfix')
args = parser.parse_args()
repo_args = utils.get_repo_args(args.repo_name)

json_path = None
with open('../all_json.json','r') as f:
    all_json = json.load(f)
    if args.repo_name in all_json.keys():
        repo_testcases = all_json[args.repo_name]
        for testcase in repo_testcases:
            if testcase['type'] == args.type:
                json_path = testcase['json_path']
                break
        if json_path is None:
            print('没有找到对应的测试用例')
            exit()
    else:
        print('Repo {} not in all_json'.format(args.repo_name))
        exit()

# args.testcase_file = os.path.join(args.output_dir, 'other_models', 'paraphase', args.repo_name, f'{args.gen_model}_paraphase.jsonl')
args.testcase_file = json_path


def complete_code(new_code, test_file, file_name, model):
    
    chat_message = f'''Below is a code file {file_name} containing a placeholder `<complete code here>`.Please analyze the provided file context and unit test information, and generate appropriate code at the `<complete code here>` location. Please output your completed code block in markdown format (```python```). The code block should only include the code at the `<completed code here>` location, without the surrounding context.
**Note**: Please ensure that your completed code block maintains the indentation of the surrounding code, meaning you need to preserve the original code's indentation.

Code file {file_name} to be completed:
```python
{new_code}
```
Corresponding unit test:
```python
{test_file}
```
'''
    res = utils.get_response(chat_message, model)
    return extract_code_loose(res), res

def remove_common_prefix(str1, str2):
    if len(str1) == 0 or len(str2) == 0:
        return str1
    try:
        if str1.startswith(str2):
            if str1[len(str2)] == '\n':
                return str1[len(str2)+1:]
            else:
                return str1[len(str2):]
        else:
            return str1
    except Exception as e:
        return str1

def remove_common_indent(text):
    """
    移除所有行与第一行相同长度的前导空格
    示例：
    输入：
        Line1
            Line2
          Line3
    输出：
    Line1
        Line2
      Line3
    """
    lines = text.splitlines(keepends=False)
    if not lines:
        return text
    
    # 计算第一行的前导空格数
    first_line = lines[0]
    indent_len = len(first_line) - len(first_line.lstrip(' '))
    processed = []
    for line in lines:
        # 计算实际可移除的空格数（取最小值，避免超出当前行长度）
        remove_count = min(indent_len, len(line) - len(line.lstrip(' ')))
        processed.append(line[remove_count:])
    
    return '\n'.join(processed)
#辅助deepseek重新生成
def is_file_empty(file_path):
    """
    检查文件是否为空（包括仅含空白字符的情况）
    :param file_path: 文件路径
    :return: True表示空文件，False表示非空或文件不存在
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取全部内容并去除首尾空白字符
            content = file.read().strip()
            
            # 如果去除空白后仍为空字符串，则认为文件为空
            return not content
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
        return False
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return False

def eval_gen_code(id, model,  repo_name, origin_file, test_path_list, prob_info, output_dir = "./"):  
    args.model = model
    args.repo_name = repo_name
    args.origin_file = origin_file
    args.repo_name = repo_name
    args.output_dir = output_dir
    args.results_dir = os.path.join(args.output_dir, 'experiments', args.type,  args.repo_name, args.model)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if os.path.exists(os.path.join(args.results_dir, f'{id}_{args.model}.py')) and not args.regenerate:
        # 如果已经存在结果文件，并且没有设置重新生成的标志，则跳过
        print(f"已经存在 {id}_{args.model}.py 的结果，跳过")
        return

    repo_args = utils.get_repo_args(args.repo_name)
    repo_path, copy_path = repo_args["repo_path"], repo_args["copy_path"]

    source_code_path = os.path.join(repo_args['repo_running_path'], origin_file)
    with open(source_code_path, 'r') as f:
        source_code = f.read().splitlines()
    
    prefix = source_code[:prob_info['key_block_start_lineno'] -1]
    suffix = source_code[prob_info['key_block_end_lineno']:]
    
    new_code = source_code[:prob_info['func_start_lineno'] -1] + prob_info['new_func_code'].splitlines() + source_code[prob_info['func_end_lineno']:]
    file_name = origin_file
    completed_code, response = complete_code('\n'.join(new_code), utils.test_path_to_str(test_path_list, repo_args['repo_path']), file_name, args.model)

    # print('\n'.join(str(x) for x in source_code[prob_info['func_start_lineno']-1:prob_info['key_block_start_lineno']-1]))
    # print(remove_common_indent(completed_code))
    completed_code = remove_common_prefix(remove_common_indent(completed_code),remove_common_indent('\n'.join(str(x) for x in source_code[prob_info['func_start_lineno']-1:prob_info['key_block_start_lineno']-1])))
    indented_completed_key_block = utils.align_indent(completed_code.splitlines(), source_code[prob_info['key_block_start_lineno']-1:prob_info['key_block_end_lineno']], prefix, suffix )
    completed_code = prefix + indented_completed_key_block + suffix

    with open(os.path.join(args.results_dir, f'{id}_{args.model}.py'),'w') as f:
        f.write('# tdd test\n' + '\n'.join(completed_code))
    with open(os.path.join(args.results_dir, f'{id}_{args.model}.prompt.py'),'w') as f:
        f.write('# code\n' + '\n'.join(new_code))
    with open(os.path.join(args.results_dir, f'{id}_{args.model}.source.py'),'w') as f:
        f.write('\n'.join(source_code)) 
    with open(os.path.join(args.results_dir, f'{id}_{args.model}.response.py'),'w') as f:
        f.write(response)

 
if __name__ == "__main__":
    testcases = []
    if not os.path.exists(args.testcase_file):
        print(f"没有找到 {args.testcase_file} 的测试用例")
        exit()
    print('测试用例来源：', args.testcase_file)
    # 读取测试用例
    with open(args.testcase_file,'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            testcases.append(data)

    for testcase in tqdm(testcases):
        id = testcase['id']
        origin_file = testcase['origin_file']
        test_path_list = testcase['test_list']
        prob_info = testcase['prob_info']
        eval_gen_code(id, args.model, args.repo_name, origin_file, test_path_list, prob_info, args.output_dir)
    