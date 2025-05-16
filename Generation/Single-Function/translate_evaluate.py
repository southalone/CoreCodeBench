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
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DeepSeekR1', help='args.model')

parser.add_argument('--repo_name', type=str, default='open-iris', help='Repository name')
parser.add_argument('--output_dir', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='regenerate/reevaluate everything')
parser.add_argument('--type', type=str, default='development', help='dev, TDD, bugfix')
args = parser.parse_args()
repo_args = utils.get_repo_args(args.repo_name)

json_path = None
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

def complete_code(prompt, model):
    chat_message = f'''Below is a code snippet containing a placeholder `<complete code here>`. Please analyze the provided context and description of the missing code to generate the appropriate code block at `<complete code here>`.
Please output the completed code block using markdown format (```python```).
**Important**: Ensure the code block you complete maintains the same indentation as the context code, meaning you need to preserve the original code's indentation.The output must exactly match the line count and structure of the input, including preserving empty lines and comment positions.
Code snippet:
```python
{prompt}
```
Please output the completed code block using markdown format (```python```). Make sure to preserve the original indentation before and after the <complete code here> placeholder. And remember don't add the signature of the function into it.
'''
    res = utils.get_response(chat_message, model)
    return extract_code(res), res


def eval_gen_code(id, model,  repo_name, origin_file, test_path_list, prob_info, output_dir = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/"):  
    args.model = model
    args.repo_name = repo_name
    args.origin_file = origin_file
    args.repo_name = repo_name
    args.output_dir = output_dir
    args.results_dir = os.path.join(args.output_dir, 'experiments_en', args.type,  args.repo_name, args.model)
    #这里用英语experiments_en
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
    chat_message = f'''Your task is to translate the following Python code comments from Chinese to English. Please follow these requirements strictly:
            
        1. Translate only the comment sections of the code without altering the code logic or structure.
        2. Ensure the translation accurately conveys the original meaning with professionalism and technical precision.
        3. Follow English expression habits for smooth and natural translated text.
        4. Preserve special symbols, formatting, and comment markers within the code.
        5. If the comments contain code snippets or examples, translate those as well.
        6. The output must exactly match the line count and structure of the input, including preserving empty lines and comment positions.
        7. Use '# Explanation of the functionality of this code segment: ' as the first line.
        8. Don't output the code again.
        9. Translate to English WITHOUT ANY additional text.
        Here is a list of terms that need uniform translation:
        - 本段代码的功能解释：-> Explanation of the functionality of this code segment:
        - 目的 -> purpose
        - 逻辑 -> logic
        - 异常 -> exceptions
        - 变量赋值 -> variable assignment

        Example (Output):
# Explanation of the functionality of this code segment: 
#1. **purpose**
#    Provides a resampling functionality for data between input and output grids. Performs interpolation on different grid structures (structured or unstructured) to generate resampled data.
#
#2. **logic**
#    - When `self.structured` is `True`, using a structured grid:
#        - Assigns the `magnitude` attribute of the input data to the `values` attribute of the interpolation object `self.inter`.
#        - Performs interpolation on `self.out_coords` using `self.inter`, resulting in the output `res`.
#        - If `self.fill_with_nearest` is `True`, fills out-of-range points with the nearest values by assigning out-of-range points in the interpolation results (indexed via `self.out_ids`) to the corresponding nearest values.
#
#    - When `self.structured` is `False`, using an unstructured grid:
#        - Input data is compressed and rearranged using `dtools.to_compressed`.
#        - The `self.inter.values` property is set to the compressed data mentioned above, ensuring the data is a contiguous array of type `np.double`.
#        - Performs interpolation on `self.out_coords` using `self.inter`, resulting in the output `res`.
#        - If `self.fill_with_nearest` is `True`, similarly fills out-of-range points with the nearest values.
#    
#    - Regardless of the grid structure, the interpolation results are formatted into the shape and order required by the output grid through `dtools.from_compressed`, considering the output mask `self.output_mask`.
#
#3. **exceptions**
#    None
#
#4. **variable assignment**
#    - `self.inter.values`: Assigned to the `magnitude` attribute of the input data when using a structured grid; assigned to the compressed data processed by `dtools.to_compressed` when using an unstructured grid.
#    - `self.out_ids`: Used to index out-of-range portions in the interpolation results when filling with nearest values.
#    - `self.fill_ids`: Used to index the array of nearest data points for correctly filling out-of-range interpolation points.

        Here are the code comments to be translated:
        {prob_info['new_func_code'].splitlines()[(prob_info['key_block_start_lineno']-prob_info['func_start_lineno']):]}

        '''
    print(prob_info['key_block_start_lineno'])
    print(prob_info['new_func_code'].splitlines()[(prob_info['key_block_start_lineno']-prob_info['func_start_lineno']):])
    print(id)
    content = extract_code(utils.get_response(chat_message,'gpt4o'))
    print(type(source_code[:prob_info['func_start_lineno'] -1]))
    #print(extract_code(content))
    #print(source_code[:prob_info['func_start_lineno'] -1] + content.split('\n') + source_code[prob_info['func_end_lineno']:])
    print(content)
    # 提取 `<complete code here>` 及其之前的所有内容
    pattern = r"^(.*?<complete code here>)"
    match = re.search(pattern, content, re.DOTALL)  # re.DOTALL 让 `.` 匹配换行符
    if match:
        content = match.group(1)  
    else:
        content = content + '\n<complete code here>'
    
    new_code = source_code[:prob_info['key_block_start_lineno'] -1] + content.split('\n') + source_code[prob_info['key_block_end_lineno']:]
    #new_code是含单个函数注释的整个文件
    completed_code, response = complete_code('\n'.join(new_code), args.model)
    print(completed_code)
    completed_code = remove_common_prefix(remove_common_indent(completed_code),remove_common_indent('\n'.join(str(x) for x in source_code[prob_info['func_start_lineno']-1:prob_info['key_block_start_lineno']-1])))
    indented_completed_key_block = utils.align_indent(completed_code.splitlines(), source_code[prob_info['key_block_start_lineno']-1:prob_info['key_block_end_lineno']], prefix, suffix )
    prefix_final = source_code[:prob_info['func_start_lineno'] -1]
    suffix_final = source_code[prob_info['func_end_lineno']:]
    completed_code = prefix + indented_completed_key_block + suffix
    #print(completed_code)

    with open(os.path.join(args.results_dir, f'{id}_{args.model}.py'),'w') as f:
        f.write('# evaluation\n' + '\n'.join(completed_code))
    with open(os.path.join(args.results_dir, f'{id}_{args.model}.prompt.py'),'w') as f:
        f.write('# code\n' + '\n'.join(new_code))
    #with open(os.path.join(args.results_dir, f'{id}_{args.model}.source.py'),'w') as f:
    #    f.write('\n'.join(source_code)) 
    #with open(os.path.join(args.results_dir, f'{id}_{args.model}.response.py'),'w') as f:
    #    f.write('\n'.join(result))

 
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
    