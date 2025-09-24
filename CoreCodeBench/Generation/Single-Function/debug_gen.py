# -*- coding: utf-8 -*-
import re
import os
import sys
import ast
# 获取当前脚本所在目录的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
import textwrap  # 用于移除额外的缩进
import json
import argparse
import subprocess
import utils
import argparse
import random
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gen_model', default='mix', help='Repository name')
parser.add_argument('--rewrite_model', default='mix', help='rewrite model')
parser.add_argument('--repo_name', default='open-iris', help='Repository name')
parser.add_argument('--output_dir', default = '')
args = parser.parse_args()

repo_args = utils.get_repo_args(args.repo_name)
args.buggy_logic_dir = os.path.join(args.output_dir, 'DEBUG', 'testcases_logic',args.repo_name, f"{args.gen_model}-{args.rewrite_model}")
if not os.path.exists(args.buggy_logic_dir):
    os.makedirs(args.buggy_logic_dir)
testcase_dir = os.path.join('output_dir', 'all_json.json')

rewrite_models = ['gpt4o', 'claude3.5', 'qwen-plus-latest', 'doubao']
gen_models = [ 'qwen2.5-7B-Coder',   'longcat-large-32K','deepseek-16B-Coder','gpt4o-mini']


def test_rewrite_models():
    for model in rewrite_models:
        print('Testing model {}'.format(model))
        print(utils.get_response('Hello! How are you today?', model))

def test_gen_models():
    for model in gen_models:
        print('Testing model {}'.format(model))
        print(utils.get_response('Hello! How are you today?', model))
        



def complete_code(prompt, model):
    chat_message = f'''下面是一段包含占位符 `<complete code here>` 的代码片段。请根据缺失代码的描述，在 `<complete code here>`处生成适当的代码块。
请使用markdown格式(```python```)输出你补全的代码块。
**注意**：请确保你补全的代码块符合上下文代码的缩进，也就是说，需要保留原来代码的缩进。
代码片段:
```python
{prompt}
```
请使用markdown格式(```python```)输出你补全的代码块。需要保留<complete code here>前后原来代码的缩进。
'''
    res = utils.get_response(chat_message, model=model)
    return utils.extract_code(res), res


def generate_buggy_code(testcase, problem, buggy_logic_response):
    def extract_buggy_logic(string):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, string, re.DOTALL)

        for content in matches:
            content = content.strip()
            if content.startswith("2. **逻辑**"):
                return content
        return None
    buggy_logic =  extract_buggy_logic(buggy_logic_response)
    if not buggy_logic:
        return False
    buggy_logic = "\n".join("# " + line for line in buggy_logic.splitlines())

    if buggy_logic:
        pattern = r"(#2\. \*\*逻辑\*\*.*?)(?=#3\. \*\*异常\*\*)"
        try:
            modified_problem = re.sub(pattern, buggy_logic, problem, flags=re.DOTALL)
        except:
            return None
        print(repo_args['repo_running_path'])
        print(testcase['origin_file'])
        source_code_path = os.path.join(repo_args['repo_running_path'], testcase['origin_file']) 
        with open(source_code_path, 'r') as f:
            source_code = f.read().splitlines()
        prompt = '\n'.join(source_code[:testcase["prob_info"]['key_block_start_lineno'] - 1]) \
                            + modified_problem + \
                '\n'.join(source_code[testcase["prob_info"]['key_block_end_lineno']:])
        if args.gen_model == 'mix':
            gen_model = random.choice(gen_models)
        else:
            gen_model = args.gen_model
        buggy_code, response = complete_code(prompt, gen_model)
        
        buggy_func_code = '\n'.join(source_code[testcase["prob_info"]['func_start_lineno'] - 1:testcase["prob_info"]['key_block_start_lineno'] - 1]) \
                                 + '\n<buggy_code_start_here>\n' +  buggy_code + '\n<buggy_code_end_here>\n' \
                                   + '\n'.join(source_code[testcase["prob_info"]['key_block_end_lineno']:testcase["prob_info"]['func_end_lineno']])
        new_testcase = testcase.copy()
        new_testcase['prob_info']= {
            'func_start_lineno': testcase['prob_info']['func_start_lineno'],
            'func_end_lineno': testcase['prob_info']['func_end_lineno'],
            'new_func_code': buggy_func_code
        }
        new_testcase['IG_base'] = None
        new_testcase['score'] = None
        new_testcase['LLM_score'] = None
        new_testcase['pytest_info'] = {
            'total_num': testcase['pytest_info']['total_num']
        }
        
        new_testcase['model_info'] = {
            'gen_model': testcase['model_info']['gen_model'] if 'gen_model' in testcase['model_info'] else 'gpt4o',
            'rewrite_model': testcase['model_info']['rewrite_model'],
            'debug_gen_model': gen_model
        }
        with open(os.path.join(args.buggy_logic_dir, '{}_buggy_logic.py'.format(testcase['id'].replace('.','_'))),'w') as f:
            f.write(prompt)
        with open(os.path.join(args.buggy_logic_dir, '{}_rewrite_model.response'.format(testcase['id'].replace('.','_'))),'w') as f:
            f.write(response)
        with open(os.path.join(args.buggy_logic_dir, '{}_buggy.py'.format(testcase['id'].replace('.','_'))),'w') as f:
            f.write(buggy_func_code)
        with open(os.path.join(args.buggy_logic_dir, '{}_source.py'.format(testcase['id'].replace('.','_'))),'w') as f:
            f.write('\n'.join(source_code[testcase["prob_info"]['func_start_lineno'] - 1: testcase["prob_info"]['func_end_lineno']]))
        new_testcase['type'] = 'debug'
        return new_testcase

    return None



def retest(testcase, tmp_repo_path):
    source_path = os.path.join(repo_args['repo_running_path'], testcase['origin_file'])
    with open(source_path, 'r') as f:
        source_code = f.read().splitlines()
    prefix = '\n'.join(source_code[:testcase['prob_info']['func_start_lineno'] - 1])
    suffix =  '\n'.join(source_code[testcase['prob_info']['func_end_lineno']:])
    new_code = prefix + testcase['prob_info']['new_func_code'].replace('<buggy_code_start_here>','').replace('<buggy_code_end_here>','') + suffix
    tmp_running_file = source_path.replace(repo_args['repo_path'], tmp_repo_path)
    with open(tmp_running_file, 'w') as f:
        f.write(new_code)
    test_path = os.path.join(tmp_repo_path, testcase['test_list'][0])
    log_path = os.path.join(args.buggy_logic_dir, 'retest_{}.log'.format(testcase['id']))
    env = os.environ.copy()
    running_path = repo_args['repo_running_path'].replace(repo_args['repo_path'], tmp_repo_path)
    env["PYTHONPATH"] =running_path
    env["http_proxy"] = "http://10.229.18.30:8412"
    env["https_proxy"] = "http://10.229.18.30:8412"
    
    pytest_command = f"export PYTHONPATH={running_path} && python -m pytest {test_path} --tb=long > {log_path} 2>&1"
    print(pytest_command)
    try:
        result = subprocess.run(pytest_command, env=env, timeout=120, shell=True)
    except subprocess.TimeoutExpired:
        print("测试超时")
        return False, None
    #恢复文件
    shutil.copy(source_path, tmp_running_file)
    if result.returncode == 0:
        print("所有测试通过或跳过")
        return False, None
    else:
        print("有测试失败或错误发生")
        passed, skipped, failed = utils.read_log(log_path)
        return True, (passed, skipped, failed)
   
def modify_logic(question, rewrite_model):
    prompt = f'''请根据以下代码解释，改写`2. **逻辑**`的部分，在其中引入一些会影响程序正确性的逻辑漏洞。要求逻辑漏洞是合理的，符合人类习惯的。
逻辑漏洞类型包括但不限于边界条件处理不当，循环或者递归条件错误，条件判断错误，变量错误等等。
请输出修改后“逻辑”项，用``` ```框出。注意，请不要显式在逻辑项中写出逻辑漏洞是什么。
待修改的解释文字：```
{question}
```
样例输出：
```
2. **逻辑**
改写后的“逻辑”项。
```
请确保漏洞难以被发现，但能够导致程序在一些情况（例如边界条件、特殊情况）下出现不符合预期的行为。
'''
    return utils.get_response(prompt, rewrite_model)


if __name__ == '__main__':
    with open(testcase_dir, 'r') as f:
        repo_testcase_dir = json.load(f)
    if args.repo_name not in repo_testcase_dir:
        exit()
    testcase_list = repo_testcase_dir[args.repo_name]
    json_path = None
    for file in testcase_list:
        if file['type'] == 'development':
            json_path = file['json_path']
            if json_path.endswith('_new.jsonl'):
                json_path = json_path.replace('_new.jsonl', '.jsonl')
            break
    if json_path is None:
        print(f"没有找到 {args.repo_name} 的测试用例")
        exit()
    
    # test_rewrite_models()
    print('Rewrite model test passed!')
    # 生成错误逻辑（rewrite）
    if not os.path.exists(os.path.join(args.buggy_logic_dir, 'buggy_logic.jsonl')):
        generated_buggy_testcases = []
    else: 
        buggy_logical_testcase = utils.load_jsonl_to_dict(os.path.join(args.buggy_logic_dir, 'buggy_logic.jsonl'), 'id')
        generated_buggy_testcases = list(buggy_logical_testcase.keys())
    
    print(json_path)
    test_file_jsonl = utils.load_jsonl_to_dict(json_path, 'id')
    
    
    with open(os.path.join(args.buggy_logic_dir, 'buggy_logic.jsonl'), 'a', encoding='utf-8') as f:
        for id, testcase in tqdm(test_file_jsonl.items()):
            if id in generated_buggy_testcases:
                continue
            
            args.copy_test_path = os.path.join(repo_args['copy_path'], testcase['test_list'][0])
            args.test_case_dir,_ = os.path.split(args.copy_test_path)
            key_block_start, key_block_end = testcase['prob_info']['key_block_start_lineno'],testcase['prob_info']['key_block_end_lineno']
            args.copy_code_path = os.path.join(repo_args['copy_running_path'], testcase['origin_file'])
            args.copy_running_path = repo_args['copy_running_path']
            args.repo_running_path = repo_args['repo_running_path']
            args.copy_path = repo_args['copy_path']
            code_block = testcase['prob_info']['new_func_code']
                
            pattern = r"# 本段代码的功能解释：(.*?)<complete code here>"
            match = re.search(pattern, code_block, re.DOTALL)
            if match:
                problem = match.group(1).strip()
            else:
                problem = None
                continue
            
            if problem:
                if args.rewrite_model =='mix':
                    rewrite_model = random.choice(rewrite_models)
                    buggy_logic_response = modify_logic(problem, rewrite_model)
                else:
                    rewrite_model = args.rewrite_model
                    buggy_logic_response = modify_logic(problem, args.rewrite_model)
                
                print(f'True logic:\n {problem}')
                print(f'Buggy Logic:\n {buggy_logic_response}')
                generated_buggy_testcases.append(id)
                buggy_testcase = testcase.copy()
                buggy_testcase['model_info'] = {
                    'rewrite_model':rewrite_model
                }
                buggy_testcase['buggy_logic_response'] = buggy_logic_response
                buggy_testcase['problem'] = problem
                f.write(json.dumps(buggy_testcase, ensure_ascii=False) + '\n')
                
                   
    assert os.path.exists(os.path.join(args.buggy_logic_dir, 'buggy_logic.jsonl'))
    # test_gen_models()
    print('Generate model test passed!')
    print('Generating buggy code....')
    buggy_testcases = utils.load_jsonl_to_dict(os.path.join(args.buggy_logic_dir, 'buggy_logic.jsonl'), 'id')
    generated_buggy_testcases = utils.load_jsonl_to_dict(os.path.join(args.buggy_logic_dir, 'first_gen_testcases.jsonl'), 'id')
    with open(os.path.join(args.buggy_logic_dir, 'first_gen_testcases.jsonl'),  'a', encoding='utf-8') as f:
        for id, testcase in buggy_testcases.items():
            if id in generated_buggy_testcases:
                continue
            problem = testcase['problem']
            buggy_logic_response = testcase['buggy_logic_response']
            buggy_testcase = generate_buggy_code(testcase, problem, buggy_logic_response)
            if buggy_testcase:
                json_line = json.dumps(buggy_testcase, ensure_ascii=False)
                f.write(json_line + '\n')
                print(buggy_testcase['id'])
    
    print('Retesting....')
    # retest path
    parent_dir = os.path.dirname(repo_args['copy_running_path'].rstrip('/\\'))
    tmp_copy_base = os.path.join(parent_dir, 'tmp')
    os.makedirs(tmp_copy_base, exist_ok=True)
    import tempfile
    import shutil
    temp_copy_path = tempfile.mkdtemp(prefix=f"{args.repo_name}_{args.gen_model}_", dir=tmp_copy_base)
    shutil.copytree(repo_args['repo_path'], temp_copy_path, dirs_exist_ok=True)
    
    if not temp_copy_path.endswith(os.sep):
        temp_copy_path += os.sep
    
    # retest
    testcase_jsonl = utils.load_jsonl_to_dict(os.path.join(args.buggy_logic_dir, 'first_gen_testcases.jsonl'), 'id') 
    valid_buggy_testcases = []
    try:
        for id, testcase in testcase_jsonl.items():
            retest_flag, scores = retest(testcase,temp_copy_path)
            if retest_flag:
                testcase['pytest_info']['base_passed_num'] = scores[0]
                valid_buggy_testcases.append(testcase)
        utils.write_list_to_jsonl(os.path.join(args.buggy_logic_dir, 'buggy_{}.jsonl'.format(args.repo_name)), valid_buggy_testcases)
    finally:
        if temp_copy_path:
            shutil.rmtree(temp_copy_path)
