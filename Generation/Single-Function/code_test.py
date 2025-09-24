# -*- coding: utf-8 -*-
from server import LocalClient
import json
import os
import shutil
from utils import generate_xlsx, read_log, extract_code
import argparse
import pandas as pd
import utils
import time


parser = argparse.ArgumentParser()
parser.add_argument('--if_comments', type=str, default='empty', help='empty or full')
parser.add_argument('--mode', type=str, default='generate', help='generate or evaluate')
parser.add_argument('--model', type=str, default='llama', help='args.model')

parser.add_argument('--repo_name', type=str, default='transformers', help='Repository name')
parser.add_argument('--file_path', type=str, default='src/transformers/', help='File path without extension')
parser.add_argument('--file_name', type=str, default='audio_utils', help='File name without extension')
parser.add_argument('--test_path', type=str, default='./Source_Copy/transformers/tests/utils/test_audio_utils.py', help='Output directory for results')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
parser.add_argument('--temp_copy_path', type=str, default='', help='Output directory for results')
parser.add_argument('--language',type=str, default='ch', help='ch/en')
parser.add_argument('--regenerate', action='store_true', help='regenerate/reevaluate everything')


args = parser.parse_args()

def complete_code(prompt, testcase_item, file_name, model):
    #     chat_message = f'''Below is a code snippet with a placeholder `<complete code here>`. Analyze the provided context and logic to generate the appropriate code to replace `<complete code here>`. Please only output the code that replaces `<complete code here>`, without any additional explanation or formatting.
    # **Caution**: 请注意你补全的代码块需要符合上下文代码的缩进，也就是说，需要保留原来代码的缩进。
    # Code snippet:
    # ```python
    # {prompt}
    # ```
    # '''
    chat_message = f'''下面是一段包含占位符 `<complete code here>` 的代码片段。请分析提供的上下文和缺失代码的描述，在 `<complete code here>`处生成适当的代码块。
请使用markdown格式(```python```)输出你补全的代码块。
**注意**：请确保你补全的代码块符合上下文代码的缩进，也就是说，需要保留原来代码的缩进。
代码片段:
```python
{prompt}
```
请使用markdown格式(```python```)输出你补全的代码块。需要保留<complete code here>前后原来代码的缩进。
'''
    res = utils.get_response(chat_message, model=model)
    return extract_code(res), res

def test_code(if_comments, mode, model, repo_name, file_path, file_name, test_path, running_path, regenerate,  output_dir = "./"):

    args.if_comments = if_comments
    args.mode = mode
    args.model = model
    args.repo_name = repo_name
    args.file_path = file_path
    args.file_name = file_name
    args.test_path = test_path
    args.output_dir = output_dir



    repo_args = utils.get_repo_args(args.repo_name)
    repo_path, copy_path = repo_args["repo_path"], repo_args["copy_path"]

    test_case_dir = os.path.join(output_dir, 'testcases', repo_name,  file_path, file_name)
    print(f"testing repo {repo_name}, repo_path {repo_path}, copy_path {copy_path}")
    print(f"testing file {file_name} in {file_path}.")
    print(f"model {model}, mode {mode}")
    print(f"test path: {test_case_dir}")

    source_code_path = os.path.join(repo_path, file_path, f'{file_name}.py')

    if args.if_comments == 'empty':
        result_dir = os.path.join(output_dir, 'results', repo_name, file_path,  file_name, args.model, "empty")
    elif args.if_comments == 'full':
        result_dir = os.path.join(output_dir, 'results', repo_name, file_path,  file_name, args.model)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # 修改Sourse_Copy中的文件
    testcases = {}
    print(os.path.join(test_case_dir, 'testcases_valid_info.jsonl'))
    
    with open(os.path.join(test_case_dir, 'testcases_valid_retest_info.jsonl'),'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            testcases[data['func']] = (data['id'], data['code'])


    with open(source_code_path, 'r') as f:
        source_code = f.read().splitlines()
    
    results_df = pd.DataFrame(columns=['test_id', 'passed', 'skipped', 'failed'])
    if args.if_comments == 'empty':
        for func, item in testcases.items():
            
            test_id, testcase = item
            
            new_func_lines = testcase['new_func_code'].splitlines()
            updated_func_lines = []
            inside_placeholder = False

            for line in new_func_lines:
                if "# 本段代码的功能解释：" in line:
                    inside_placeholder = True
                elif "<complete code here>" in line:
                    inside_placeholder = False
                    updated_func_lines.append("<complete code here>")
                elif not inside_placeholder:
                    updated_func_lines.append(line)

            prompt = source_code[:testcase['func_start_lineno'] - 1] + updated_func_lines + source_code[testcase['func_end_lineno']:]
            print('FUNC',func)
            
            file_exists = os.path.exists(os.path.join(result_dir,f'''{func}_completed_{args.model}.py'''))
            
            #if file_exists and not regenerate:
            #    continue # 跳过

            if not file_exists or args.mode == 'generate':
                completed_key_block, response = complete_code('\n'.join(prompt), testcase, func,args.model)
                print(completed_key_block)
                
                # 对齐缩进
                prefix = source_code[:testcase['key_block_start_lineno'] -1]
                suffix = source_code[testcase['key_block_end_lineno']:]
                indented_completed_key_block = utils.align_indent(completed_key_block.splitlines(), source_code[testcase['key_block_start_lineno']-1:testcase['key_block_end_lineno']], prefix, suffix )
                compeletd_code = prefix + indented_completed_key_block + suffix

                
                with open(os.path.join(result_dir,f'''{func}_completed_{args.model}.py'''),'w') as f:
                    f.write('#test_copy\n' + '\n'.join(compeletd_code))
                with open(os.path.join(result_dir,f'''{func}_completed_{args.model}.prompt.py'''),'w') as f:
                    f.write('\n'.join(prompt))
                with open(os.path.join(result_dir, f'''{func}_completed_{args.model}.source.py'''),'w') as f:
                    f.write('\n'.join(source_code))
                with open(os.path.join(result_dir, f'''{func}_response_{args.model}.response.py'''),'w') as f:
                    f.write(response)
                print(f'============result is written to {result_dir}============')

            if args.mode == 'evaluate':
                if not regenerate and os.path.exists(os.path.join(result_dir, f'''results.xlsx''')):
                    return 
                shutil.copy(os.path.join(result_dir,f'''{func}_completed_{args.model}.py'''), source_code_path.replace(repo_args['repo_path'], running_path))
                os.chdir(os.path.join(running_path,repo_args['relative_running_path']))
                print(os.path.join(running_path,repo_args['relative_running_path']))

                BASH = f'''PYTHONPATH={os.path.join(running_path,repo_args['relative_running_path'])} pytest {test_path} --tb=long > {result_dir}/{func}_{args.model}_result.log''' 
                os.system(BASH)
                if os.path.exists(os.path.join(result_dir, f'''{func}_{args.model}_result.log''')):
                    passed, skipped, failed = read_log(os.path.join(result_dir, f'''{func}_{args.model}_result.log'''))
                else:
                    passed, skipped, failed = [0,0,0]
                results_df = results_df._append({'test_id': [test_id], 'passed': [passed], 'skipped': [skipped], 'failed': [failed]}, ignore_index=True)
                shutil.copy(source_code_path, source_code_path.replace(source_code_path, source_code_path.replace(repo_args['repo_path'], running_path))) # 复原文件
    
    elif args.if_comments == 'full':
        for func, item in testcases.items():
            test_id, testcase = item
            prompt = source_code[:testcase['func_start_lineno'] -1] + testcase['new_func_code'].splitlines() + source_code[testcase['func_end_lineno']:]        

            print('FUNC',func)
            file_exists = os.path.exists(os.path.join(result_dir,f'''{func}_completed_{args.model}.py'''))
            if file_exists and not regenerate:
                continue # 跳过
            if not file_exists or args.mode == 'generate':
                completed_key_block, response = complete_code('\n'.join(prompt), testcase, func, args.model)
                # 对齐缩进
                prefix = source_code[:testcase['key_block_start_lineno'] -1]
                suffix = source_code[testcase['key_block_end_lineno']:]
                print(completed_key_block)
                indented_completed_key_block = utils.align_indent(completed_key_block.splitlines(), source_code[testcase['key_block_start_lineno']-1:testcase['key_block_end_lineno']], prefix, suffix )
                compeletd_code = prefix + indented_completed_key_block + suffix

                print(f'''key_block_start_lineno: {testcase['key_block_start_lineno']},key_block_end_lineno:{testcase['key_block_end_lineno']}''' )
                print(f'''func_start_lino:{testcase['func_start_lineno']},func_end_lineno:{testcase['func_end_lineno']}''')

                with open(os.path.join(result_dir,f'''{func}_completed_{args.model}.py'''),'w') as f:
                    f.write('#test_copy\n' + '\n'.join(compeletd_code))
                with open(os.path.join(result_dir,f'''{func}_completed_{args.model}.prompt.py'''),'w') as f:
                    f.write('\n'.join(prompt))
                with open(os.path.join(result_dir, f'''{func}_completed_{args.model}.source.py'''),'w') as f:
                    f.write('\n'.join(source_code))
                with open(os.path.join(result_dir, f'''{func}_response_{args.model}.response.py'''),'w') as f:
                    f.write(response)
                print(f'============result is written to {result_dir}============')
            
            if args.mode == 'evaluate':
                if not regenerate and os.path.exists(os.path.join(result_dir, f'''results.xlsx''')):
                    return 
                shutil.copy(os.path.join(result_dir,f'''{func}_completed_{args.model}.py'''), source_code_path.replace(repo_args['repo_path'], running_path))
                os.chdir(os.path.join(running_path,repo_args['relative_running_path']))
                print(os.path.join(running_path,repo_args['relative_running_path']))

                BASH = f'''PYTHONPATH={os.path.join(running_path,repo_args['relative_running_path'])} pytest {test_path} --tb=long > {result_dir}/{func}_{args.model}_result.log''' 
                exit_code = os.system(BASH)
                if os.path.exists(os.path.join(result_dir, f'''{func}_{args.model}_result.log''')):
                    passed, skipped, failed = read_log(os.path.join(result_dir, f'''{func}_{args.model}_result.log'''))
                else:
                    passed, skipped, failed = [0,0,0]
                results_df = results_df._append({'test_id': [test_id], 'passed': [passed], 'skipped': [skipped], 'failed': [failed]}, ignore_index=True)
                # 恢复源文件
                
                shutil.copy(source_code_path, source_code_path.replace(source_code_path, source_code_path.replace(repo_args['repo_path'], running_path))) # 复原文件

    if args.mode == 'evaluate':
        results_df.to_excel(os.path.join(result_dir, f'''results.xlsx'''), index=False)



if __name__ == "__main__":
    running_path = "./Source_Copy/finam"
    test_code(args.if_comments, args.mode, args.model, args.repo_name, args.file_path, args.file_name, args.test_path,args.regenerate, running_path, args.output_dir)
    