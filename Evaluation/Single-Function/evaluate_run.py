# -*- coding: utf-8 -*-
import json
import os
import shutil
import sys
from utils import generate_xlsx, read_log, extract_code
import argparse
import pandas as pd
import utils
import time

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='o1mini', help='args.model')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true')
parser.add_argument('--type', type=str, default='development', help='dev, TDD, bugfix')
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
args = parser.parse_args()

repo_args = utils.get_repo_args(args.repo_name)


def evaluate_gen_code(id, model, repo_name, origin_file, test_path_list, prob_info, tmp_repo_path, output_dir):
    repo_args = utils.get_repo_args(repo_name)
    args.model = model
    args.repo_name = repo_name
    args.origin_file = origin_file
    args.repo_name = repo_name
    args.output_dir = output_dir
    args.results_dir = os.path.join(args.output_dir, 'experiments', args.type, args.repo_name, args.model)

    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if not os.path.exists(os.path.join(args.results_dir, f'{id}_{args.model}.py')):
        print(f'不存在 {id}_{args.model}.py 的结果，生成中.....')
        from evaluate import eval_gen_code
        eval_gen_code(id, model, repo_name, origin_file, test_path_list, prob_info, output_dir)
    
    running_path = repo_args['repo_running_path'].replace(repo_args['repo_path'], tmp_repo_path)
    print('tmp_repo_path:', tmp_repo_path)
    print('running_path:', running_path)
    source_code_path = os.path.join(running_path, origin_file)
    completed_code_path = os.path.join(args.results_dir, f'{id}_{args.model}.py')
    passed_list, skipped_list, failed_list = [], [], []
    # 复制文件
    shutil.copy(completed_code_path, source_code_path)
    
    for test_path in test_path_list:
        test_file_path = os.path.join(tmp_repo_path, test_path)
        
        os.chdir(running_path)
        BASH = f'''PYTHONPATH={running_path} timeout 120 pytest {test_file_path} --tb=long > {args.results_dir}/{id}.log''' 
        print('Running....', BASH)
        os.system(BASH)
        passed, skipped, failed = read_log(os.path.join(args.results_dir, f'''{id}.log'''))
        passed_list.append(passed) 
        skipped_list.append(skipped)
        failed_list.append(failed)
      

    # 恢复文件
    shutil.copy(os.path.join(repo_args['repo_running_path'], origin_file),source_code_path)
    
    return {
        'id': id,
        'passed': passed_list,
        'skipped': skipped_list,
        'failed': failed_list
    }

def calc_pass_rate(result_jsonl, testcases):
    # 计算通过率
        pass_all_list = []
        pass_rate_list = []
        
        assert os.path.exists(result_jsonl)
        results = utils.load_jsonl_to_dict(result_jsonl, 'id')
        for testcase in testcases:
            #assert testcase['id'] in results
            if testcase['id'] in results:
                pass_all_list.append(sum(results[testcase['id']]['passed']) == testcase['pytest_info']['total_num'])
                pass_rate = max(0, (sum(results[testcase['id']]['passed'])-testcase['pytest_info']['base_passed_num'])/(testcase['pytest_info']['total_num']-testcase['pytest_info']['base_passed_num']) )
                pass_rate_list.append(pass_rate)
            
        print('Model {}: pass_all = {}, pass_rate = {}'.format(args.model, sum(pass_all_list)/ len(pass_all_list), sum(pass_rate_list)/ len(pass_rate_list)))
        avg_pass_all = sum(pass_all_list) / len(pass_all_list) if pass_all_list else 0.0
        avg_pass_rate = sum(pass_rate_list) / len(pass_rate_list) if pass_rate_list else 0.0
        stats = {
        'model': args.model,
        'pass_all': avg_pass_all,
        'pass_rate': avg_pass_rate,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'num_testcases': len(pass_all_list)
    }

        
        # 写入JSONL文件（自动创建目录）
        stats_dir = os.path.dirname(os.path.dirname(result_jsonl))
        os.makedirs(stats_dir, exist_ok=True)
        stats_file = os.path.join(stats_dir, 'results.jsonl')
        
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
        


if __name__ == "__main__":
    testcases = []
    
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
    if not os.path.exists(args.testcase_file):
        print(f"没有找到 {args.testcase_file} 的结果，跳过")
        sys.exit(0)
    # 读取测试用例
    with open(args.testcase_file,'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            testcases.append(data)
    

    # 测试
    if not os.path.exists(os.path.join(args.output_dir, 'experiments', args.type, args.repo_name, args.model)):
        os.makedirs(os.path.join(args.output_dir, 'experiments',args.type, args.repo_name, args.model))
    
   
    result_file = os.path.join(args.output_dir, 'experiments', args.type, args.repo_name, args.model, f'results.jsonl')
    print(result_file)
    if os.path.exists(result_file) and not args.regenerate:
    #if args.regenerate:
            # 已经算过分了
        print('已经存在结果文件')
        calc_pass_rate(result_file, testcases)
    else:
            #  临时测试文件
        temp_dir = os.path.join(args.output_dir, 'tmp_source_code')
        os.makedirs(temp_dir, exist_ok=True)
        import tempfile
        temp_copy_path = tempfile.mkdtemp(prefix=f'{args.repo_name}_EVAL_', dir=temp_dir)
        shutil.copytree(repo_args['repo_path'], temp_copy_path, dirs_exist_ok=True)
        if not temp_copy_path.endswith(os.sep):
            temp_copy_path += os.sep
        try:
            with open(result_file, 'w') as f:
                for testcase in tqdm(testcases):
                    id = testcase['id']
                    origin_file = testcase['origin_file']
                    test_path_list = testcase['test_list']
                    prob_info = testcase['prob_info']
                    res = evaluate_gen_code(id, args.model, args.repo_name, origin_file, test_path_list, prob_info, temp_copy_path, args.output_dir)
                    
                    f.write(json.dumps(res))
                    f.write('\n')
                calc_pass_rate(result_file, testcases)
        
        finally:
            if temp_copy_path:
                shutil.rmtree(temp_copy_path)    
