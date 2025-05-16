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

parser.add_argument('--repo_name', type=str, default='open-iris', help='Repository name')
parser.add_argument('--output_dir', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='regenerate/reevaluate everything')
parser.add_argument('--type', type=str, default='development', help='dev, TDD, bugfix')
args = parser.parse_args()
repo_args = utils.get_repo_args(args.repo_name)

json_path = None
#with open('../all_json.json','r') as f:
with open('all_json.json','r') as f:
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
# 定位最后一个 "new" 并去除
dir_path = os.path.dirname(json_path)
filename = os.path.basename(json_path)
last_new_index = filename.rfind('_new')  # 查找最后一个 new 的索引
if last_new_index != -1:
    # 计算要保留的范围: 0到new开头 + new结尾到字符串末尾
    new_filename = filename[:last_new_index] + filename[last_new_index+4:]
    new_path = os.path.join(dir_path, new_filename)
else:
    new_path = json_path  # 如果没有找到保持原样
args.testcase_file = new_path


def remove_common_prefix(str1, str2):
    if str1.startswith(str2):
        if str1[len(str2)] == '\n':
            return str1[len(str2)+1:]
        else:
            return str1[len(str2):]
    else:
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

def complete_code(new_code, model):
    chat_message = f'''下面是一段包含占位符 `<complete code here>` 的代码片段。请分析提供的上下文和缺失代码的描述，在 `<complete code here>`处生成适当的代码块。
请使用markdown格式(```python```)输出你补全的代码块。
**注意**：请确保你补全的代码块符合上下文代码的缩进，也就是说，需要保留原来代码的缩进。
代码片段:
```python
{new_code}
```
请使用markdown格式(```python```)输出你补全的代码块。需要保留<complete code here>前后原来代码的缩进。
'''
    res = utils.get_response(chat_message, model)
    return extract_code(res), res


def eval_gen_code(id, model,  repo_name, origin_file, test_path_list, prob_info, output_dir = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/"):  
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
    
    completed_code, response = complete_code('\n'.join(new_code), args.model)
    completed_code = remove_common_prefix(remove_common_indent(completed_code),remove_common_indent('\n'.join(str(x) for x in source_code[prob_info['func_start_lineno']-1:prob_info['key_block_start_lineno']-1])))
    indented_completed_key_block = utils.align_indent(completed_code.splitlines(), source_code[prob_info['key_block_start_lineno']-1:prob_info['key_block_end_lineno']], prefix, suffix )
    completed_code = prefix + indented_completed_key_block + suffix

    with open(os.path.join(args.results_dir, f'{id}_{args.model}.py'),'w') as f:
        f.write('# evaluation\n' + '\n'.join(completed_code))
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
    