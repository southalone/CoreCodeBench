# -*- coding: utf-8 -*-
import json
import os
import shutil
import sys
# 获取当前脚本所在目录的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
from utils import generate_xlsx, read_log, extract_code
import argparse
import pandas as pd
import utils
import time
from debug_code_test_logic import test_gen_code
from tqdm import tqdm
print("Command line arguments:", sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama', help='args.model')
parser.add_argument('--repo_name', type=str, default='open-iris', help='Repository name')
parser.add_argument('--gen_model', type=str, default='mix', help='gen model')
parser.add_argument('--rewrite_model', type=str, default='mix', help='rewrite model')
parser.add_argument('--output_dir', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='regenerate/reevaluate everything')
parser.add_argument('--type', type=str, default='bugfix', help='dev, TDD, bugfix')

args = parser.parse_args()
print(args)
repo_args = utils.get_repo_args(args.repo_name)
args.debug_output_dir = os.path.join(args.output_dir, 'experiments', args.type)
args.debug_testcases_dir = os.path.join(args.output_dir, 'DEBUG', 'testcases_logic')

args.debug_results_dir = args.debug_output_dir


def evaluate_gen_code(id, model, repo_name, file_path, file_name, test_path_list, prob_info, tmp_repo_path, output_dir = "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/"):
    repo_args = utils.get_repo_args(repo_name)
    debug_output_dir = os.path.join(args.output_dir, 'DEBUG')
    debug_result_dir = args.debug_results_dir
    running_path = repo_args['repo_running_path'].replace(repo_args['repo_path'], tmp_repo_path)
    source_code_path = os.path.join(running_path, file_path, f'{file_name}.py')
    result_dir = os.path.join(debug_result_dir, repo_name, args.model)
    completed_code_path = os.path.join(result_dir, f'{id}_completed_code.py')
    if not os.path.exists(completed_code_path):
        test_gen_code(id, model, repo_name, file_path, file_name, test_path_list, prob_info, output_dir)
    passed_list, skipped_list, failed_list = [], [], []
    # 复制文件
    
    shutil.copy(completed_code_path, source_code_path)
    for test_path in test_path_list:
        test_file_path = os.path.join(tmp_repo_path,test_path)
        os.chdir(running_path)
        test_file = test_path.split('.')[-1]
        BASH = f'''PYTHONPATH={running_path} timeout 120 pytest {test_file_path} --tb=long > {result_dir}/{id}_{test_file}.log''' 
        print('Running....', BASH)
        os.system(BASH)
        passed, skipped, failed = read_log(os.path.join(result_dir, f'''{id}_{test_file}.log'''))
        passed_list.append(passed) 
        skipped_list.append(skipped)
        failed_list.append(failed)
      

    # 恢复文件
    shutil.copy(os.path.join(repo_args['repo_running_path'], file_path, f'{file_name}.py'),source_code_path)
    

    return {
        'id': id,
        'passed': passed_list,
        'skipped': skipped_list,
        'failed': failed_list
    }
       

if __name__ == "__main__":
    testcases = []
    with open(os.path.join(args.debug_testcases_dir,  args.repo_name, f"{args.gen_model}-{args.rewrite_model}", f'buggy_{args.repo_name}_new.jsonl'),'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            testcases.append(data)
    # # 临时测试文件
    temp_dir = os.path.join(args.output_dir, 'tmp_source_code')
    os.makedirs(temp_dir, exist_ok=True)
    import tempfile
    temp_copy_path = tempfile.mkdtemp(prefix=f'{args.repo_name}_DEBUGEVAL_', dir=temp_dir)
    shutil.copytree(repo_args['repo_path'], temp_copy_path, dirs_exist_ok=True)
    if not temp_copy_path.endswith(os.sep):
        temp_copy_path += os.sep
    tmp_running_path = temp_copy_path

    results_dir = os.path.join(args.debug_results_dir, args.repo_name, args.model, f'{args.repo_name}_results.jsonl' )
    testcases_tested = {}
    if os.path.exists(results_dir):
        testcases_tested = utils.load_jsonl_to_dict(results_dir, 'id')
    try:
        with open(os.path.join(args.debug_results_dir, args.repo_name, args.model, f'{args.repo_name}_results.jsonl' ), 'a') as f:
            for testcase in tqdm(testcases):
                id = testcase['id']
                if id in testcases_tested and args.regenerate == False:
                    print('testcase {} is tested!'.format(id))
                    continue
                file_path, filename_with_extension = os.path.split(testcase['origin_file'])
                file_name, _ = os.path.splitext(filename_with_extension)
                test_path_list = testcase['test_list']
                prob_info = testcase['prob_info']
                res = evaluate_gen_code(id, args.model, args.repo_name, file_path, file_name, test_path_list, prob_info, tmp_running_path, args.output_dir)
                f.write(json.dumps(res))
                f.write('\n')
        
    finally:
        if tmp_running_path:
            shutil.rmtree(tmp_running_path)    
