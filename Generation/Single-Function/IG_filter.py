# -*- coding: utf-8 -*-
import sys
import os
import ast
import subprocess
import json
import os
import shutil
from utils import generate_xlsx, read_log, get_response
import argparse
import pandas as pd
import utils
import time



parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='transformers', help='Repository name')
parser.add_argument('--output_dir', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='regenerate the unittest')
parser.add_argument('--language', type=str, default='cn', help='cn/en')

args = parser.parse_args()

copy_root_path = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/Source_Copy/'
repo_args = utils.get_repo_args(args.repo_name)
#args.repo_name = repo_args["repo_name_real"]
args.repo_path = repo_args["repo_path"]
args.copy_path = repo_args["copy_path"]
args.mapping_path = os.path.join(args.output_dir, 'testcases', args.repo_name, 'output_testcase_mapping_valid.jsonl')
args.testcase_path = os.path.join(args.output_dir, 'testcases', args.repo_name)
args.test_path = os.path.join(args.output_dir, 'results', args.repo_name)
args.running_path_origin = repo_args['repo_running_path']
args.running_path_copy = args.running_path_origin.replace(args.repo_path, args.copy_path)
print(args.repo_path)
print(args.copy_path)
print(args.running_path_origin)
import os
import pandas as pd

# 定义目标文件夹名称
target_models = ['claude3.5', 'doubao', 'gpt4o', 'o1mini', 'llama', 'deepseekr1', 'qwen-plus-latest','qwen3']
# consider_models = ['claude3.5', 'doubao','gpt4o', 'llama','qwen3']#o1mini被封
calc_models =  ['claude3.5', 'doubao','gpt4o','qwen-plus-latest']
consider_models = ['claude3.5', 'doubao','gpt4o', 'qwen-plus-latest']

def generate_result(folder_paths):
# 2. 处理每个文件夹的 results.xlsx
    result_dict = {}
    for model_name, paths in folder_paths.items():
        if model_name in consider_models:
            for path in paths:
                results_file = os.path.join(path, 'results.xlsx')
                if os.path.exists(results_file):
                    #print(results_file)
                    df = pd.read_excel(results_file,engine='openpyxl')
                    for _, row in df.iterrows():
                        test_id = row['test_id'].strip('[]').strip('\'')
                        passed = int(row['passed'].strip('[]'))
                        skipped = int(row['skipped'].strip('[]'))
                        failed = int(row['failed'].strip('[]'))
                        if test_id not in result_dict:
                            result_dict[test_id] = {}
                        result_dict[test_id][model_name] = {'fill_result': [passed, skipped, failed]}

    # 3. 处理每个文件夹下的 empty 文件夹中的 results.xlsx
    for model_name, paths in folder_paths.items():
        if model_name in consider_models:
            for path in paths:
                empty_folder = os.path.join(path, 'empty')
                results_file = os.path.join(empty_folder, 'results.xlsx')
                if os.path.exists(results_file):
                    print(results_file)
                    df = pd.read_excel(results_file)
                    for _, row in df.iterrows():
                        test_id = row['test_id'].strip('[]').strip('\'')
                        passed = int(row['passed'].strip('[]'))
                        skipped = int(row['skipped'].strip('[]'))
                        failed = int(row['failed'].strip('[]'))
                        if test_id not in result_dict:
                            result_dict[test_id] = {}
                        # 将 empty 文件夹的结果存储为 'empty_result'
                        if model_name not in result_dict[test_id]:
                            result_dict[test_id][model_name] = {}
                        result_dict[test_id][model_name]['empty_result'] = [passed, skipped, failed]

    
    # 定义保存文件的路径
    if args.language == 'cn':
        output_file = os.path.join(args.test_path, 'test_results.json')
    else:
        output_file = os.path.join(args.test_path, 'test_results_translation.json')

    # 将 result_dict 保存为 JSON Lines 格式
    with open(output_file, 'w') as f:
        json_str = json.dumps(result_dict, indent=4)
        f.write(json_str)
        
def write_pytest_info():
    # 1. 读取 IG_scores.json 文件
    results_file = os.path.join(args.test_path, 'test_results.json')
    with open(results_file, 'r') as f:
        result_dict = json.load(f)

    # 合并 args.testcase_path 下的所有 testcases_valid_info.jsonl 文件
    testcase_list = []
    for root, dirs, files in os.walk(args.testcase_path):  # 替换 args.testcase_path 为正确的路径
        for file in files:
            #if file.endswith('testcases_valid_info.jsonl'):
            if file.endswith('testcases_valid_retest_info.jsonl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        testcase = json.loads(line)
                        testcase_list.append(testcase)
    # 2. 筛选 testcases
    filtered_testcases = []

    for testcase in testcase_list:
        test_id = testcase['id']  # 假设 testcase 中有 'test_id' 字段
        if test_id not in result_dict:
            print(f"Warning: Test ID {test_id} not found in testcases_valid_retest_info.")
            continue
        filtered_testcases.append(testcase)
    # 3. 保存到新的 JSON Lines 文件中
    if args.language == 'cn':
        output_file = os.path.join(args.test_path, f'{args.repo_name}_full.jsonl')
    elif args.language == 'en':
        output_file = os.path.join(args.test_path, f'{args.repo_name}_full_translation.jsonl')
    with open(output_file, 'w',encoding='utf-8') as f:
        for testcase in filtered_testcases:
            json_line = json.dumps(testcase,ensure_ascii=False)
            f.write(json_line + '\n')
    if args.language == 'cn':
        output_file = os.path.join(args.test_path, f'{args.repo_name}.jsonl')
    elif args.language == 'en':
        output_file = os.path.join(args.test_path, f'{args.repo_name}_translation.jsonl')

    mapping = utils.load_jsonl_to_dict(args.mapping_path, 'origin_file')
    print(args.mapping_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        for testcase in filtered_testcases:
            assert testcase['id'].endswith(testcase['func'])
            origin_path = testcase['id'][:-len(testcase['func'])-1].replace('.', '/') + '.py' 
            if origin_path.startswith('langchain_core'):
                origin_path = testcase['id'][:-len(testcase['func'])-1].replace('langchain_core','langchain',1).replace('.', '/') + '.py'
            origin_path = os.path.relpath(mapping[os.path.join(copy_root_path, origin_path)]['origin_file'],args.running_path_copy)          
            if os.path.relpath(mapping[os.path.join(args.running_path_copy, origin_path)]['origin_file'],copy_root_path).startswith('langchain_core'):
                origin_path = testcase['id'][:-len(testcase['func'])-1].replace('langchain_core','langchain',1).replace('.', '/') + '.py'
                mapping_i = mapping[os.path.join(args.running_path_copy, origin_path)]
            else:
                mapping_i = mapping[os.path.join(args.running_path_copy, origin_path)]
            test_path = mapping_i['test_file']
            pytest_num = mapping_i['pytest'] if 'pytest' in mapping_i else {'passed': None}
            test_path_rel = os.path.relpath(test_path, args.copy_path)
            LLM_score = testcase['LLM_score'] if 'LLM_score' in testcase else {"readability_score": None, "accuracy_score": None, "completeness_score": None}
            unittest = testcase['unittest'] if 'unittest' in testcase else 0
            gen_model = testcase['gen_model'] if 'gen_model' in testcase else 'gpt4o'
            #is_difficult_input = 'difficult' if is_difficult else 'normal' 
            origin_path = os.path.relpath(mapping[os.path.join(args.running_path_copy, origin_path)]['origin_file'],args.running_path_copy)
            output_dict = {
                'id': testcase['id'],
                'project': args.repo_name,
                'func': testcase['func'],
                'origin_file': origin_path,
                'test_list':[test_path_rel],
                'prob_info': {
                    'func_start_lineno': testcase['code']['func_start_lineno'],
                    'func_end_lineno': testcase['code']['func_end_lineno'],
                    'key_block_start_lineno': testcase['code']['key_block_start_lineno'],
                    'key_block_end_lineno': testcase['code']['key_block_end_lineno'],
                    'new_func_code': testcase['code']['new_func_code']
                },
                'pytest_info':{
                    'total_num': pytest_num['passed'],
                    'base_passed_num': unittest, 
                },
                'score': {"readability_score": None, "accuracy_score": None, "completeness_score": None},
                "LLM_score": LLM_score,
                'type': "Development",
                'language':'Python',
                'gen_model': gen_model
                
            }
            
            json_line = json.dumps(output_dict,ensure_ascii=False)

            f.write(json_line + '\n')

    print(f"Filtered {len(filtered_testcases)}/{len(testcase_list)} testcases saved to {output_file}")
def filter_testcases():
    # 1. 读取 IG_scores.json 文件
    ig_scores_file = os.path.join(args.test_path, 'IG_scores.json')
    with open(ig_scores_file, 'r') as f:
        IG_score_dict = json.load(f)

    # 合并 args.testcase_path 下的所有 testcases_valid_info.jsonl 文件
    testcase_list = []
    for root, dirs, files in os.walk(args.testcase_path):  # 替换 args.testcase_path 为正确的路径
        for file in files:
            if file.endswith('testcases_valid_retest_info.jsonl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        testcase = json.loads(line)
                        testcase_list.append(testcase)

    # 2. 筛选 testcases
    filtered_testcases = []

    for testcase in testcase_list:
        test_id = testcase['id']  # 假设 testcase 中有 'test_id' 字段
        if test_id not in IG_score_dict:
            print(f"Warning: Test ID {test_id} not found in IG scores.")
            continue
        model_scores = IG_score_dict[test_id]
        model_IG_base = [v['IG_base'] for k,v in model_scores.items()]
        is_simple = [v['is_simple'] for k, v in model_scores.items()]
        is_difficult = all(not x for x in is_simple)
        # 检查所有有记录的模型的 IG base score
        if not is_difficult:
            if all(score <= 0 for score in model_IG_base):
                continue  # 筛掉所有 IG base score <= 0 的记录
            if all(is_simple):
                continue

        LLM_score = testcase['LLM_score'] if 'LLM_score' in testcase else {"readability_score": None, "accuracy_score": None, "completeness_score": None}
        if all(score is not None for score in LLM_score.values()):
                total_score = sum(LLM_score.values())
                if total_score < 5:
                    continue
         # 留下的记录添加到结果列表中
        filtered_testcases.append(testcase)
    # 3. 保存到新的 JSON Lines 文件中
    if args.language == 'cn':
        output_file = os.path.join(args.test_path, f'{args.repo_name}_full.jsonl')
    elif args.language == 'en':
        output_file = os.path.join(args.test_path, f'{args.repo_name}_full_translation.jsonl')
    with open(output_file, 'w',encoding='utf-8') as f:
        for testcase in filtered_testcases:
            json_line = json.dumps(testcase,ensure_ascii=False)
            f.write(json_line + '\n')

    if args.language == 'cn':
        output_file = os.path.join(args.test_path, f'{args.repo_name}.jsonl')
    elif args.language == 'en':
        output_file = os.path.join(args.test_path, f'{args.repo_name}_translation.jsonl')

    mapping = utils.load_jsonl_to_dict(args.mapping_path, 'origin_file')
    with open(output_file, 'w', encoding='utf-8') as f:
        for testcase in filtered_testcases:
            assert testcase['id'].endswith(testcase['func'])
            origin_path = testcase['id'][:-len(testcase['func'])-1].replace('.', '/') + '.py' 
            if origin_path.startswith('langchain_core'):
                origin_path = testcase['id'][:-len(testcase['func'])-1].replace('langchain_core','langchain',1).replace('.', '/') + '.py'
            origin_path = os.path.relpath(mapping[os.path.join(copy_root_path, origin_path)]['origin_file'],args.running_path_copy)          
            if os.path.relpath(mapping[os.path.join(args.running_path_copy, origin_path)]['origin_file'],copy_root_path).startswith('langchain_core'):
                origin_path = testcase['id'][:-len(testcase['func'])-1].replace('langchain_core','langchain',1).replace('.', '/') + '.py'
                mapping_i = mapping[os.path.join(args.running_path_copy, origin_path)]
            else:
                mapping_i = mapping[os.path.join(args.running_path_copy, origin_path)]
            test_path = mapping_i['test_file']
            pytest_num = mapping_i['pytest'] if 'pytest' in mapping_i else {'passed': None}
            #test_path_rel = os.path.relpath(test_path, args.running_path_copy)
            test_path_rel = os.path.relpath(test_path, args.copy_path)
            LLM_score = testcase['LLM_score'] if 'LLM_score' in testcase else {"readability_score": None, "accuracy_score": None, "completeness_score": None}
            unittest = testcase['unittest'] if 'unittest' in testcase else 0
            gen_model = testcase['gen_model'] if 'gen_model' in testcase else 'gpt4o'
            test_id = testcase['id']
            model_scores = IG_score_dict[test_id]
            model_IG_base = [v['IG_base'] for k,v in model_scores.items()]
            is_simple = [v['is_simple'] for k, v in model_scores.items()]
            is_difficult = all(not x for x in is_simple)
            is_difficult_input = 'difficult' if is_difficult else 'normal' 
            output_dict = {
                'id': testcase['id'],
                'project': args.repo_name,
                'func': testcase['func'],
                'origin_file': origin_path,
                'test_list':[test_path_rel],
                'prob_info': {
                    'func_start_lineno': testcase['code']['func_start_lineno'],
                    'func_end_lineno': testcase['code']['func_end_lineno'],
                    'key_block_start_lineno': testcase['code']['key_block_start_lineno'],
                    'key_block_end_lineno': testcase['code']['key_block_end_lineno'],
                    'new_func_code': testcase['code']['new_func_code']
                },
                'pytest_info':{
                    'total_num': pytest_num['passed'],
                    'base_passed_num': unittest, 
                },
                'score': {"readability_score": None, "accuracy_score": None, "completeness_score": None},
                "LLM_score": LLM_score,
                'type': "Development",
                'language':'Python',
                'gen_model':gen_model,
                'is_difficult': is_difficult_input
                
            }
            json_line = json.dumps(output_dict,ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"Filtered {len(filtered_testcases)}/{len(testcase_list)} testcases saved to {output_file}")

def calc_IG_score():
    # 1. 从 test_results.json 读取结果
    results_file = os.path.join(args.test_path, 'test_results.json')
    with open(results_file, 'r') as f:
        result_data = json.load(f)  # 读取整个 JSON 文件为一个字典

    # 初始化存储 IG 得分的字典
    IG_scores = {}

    # 2. 遍历每个测试点
    for test_id, model_results in result_data.items():
        IG_scores[test_id] = {}

        # 从 {repo_name}.jsonl 文件中获取 base_passed_num 和 total_num
        jsonl_file = os.path.join(args.test_path, f"{args.repo_name}.jsonl")
        base_passed_num = 0
        total_num = 0
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'pytest_info' in data:
                    pytest_info = data['pytest_info']
                    total_num = pytest_info.get('total_num', 0)
                    base_passed_num = pytest_info.get('base_passed_num', 0)
                if data['id'] == test_id:
                    break  # 假设只有一行包含 pytest_info
        # 计算每个模型的 IG base 得分
        for model, results in model_results.items():
            #print(test_id, model, results)
            fill_passed, fill_skipped, fill_failed = results['fill_result']
            
            # 计算 fill_success_rate，考虑 base_passed_num
            print(total_num)
            denominator = total_num - base_passed_num
            if denominator > 0:
                fill_success_rate = max(0, (fill_passed - base_passed_num) / denominator)
            else:
                fill_success_rate = 0

            # 计算 empty_success_rate，假设 empty_result 不需要考虑 base_passed_num
            empty_passed, empty_skipped, empty_failed = results['empty_result']
            if denominator > 0:
                empty_success_rate = max(0, (empty_passed - base_passed_num) / denominator)
            else:
                empty_success_rate = 0
            # 计算 IG base 得分
            IG_base_score = fill_success_rate - empty_success_rate
            IG_scores[test_id][model] = {'IG_base': IG_base_score, 'IG_decay': None, 'is_simple': fill_success_rate == 1}

    # 3. 将 IG 得分结果记录到 IG_scores.json 中
    output_file = os.path.join(args.test_path, 'IG_scores.json')
    with open(output_file, 'w') as f:
        json.dump(IG_scores, f, indent=4)
    print(f"IG scores saved to {output_file}")

def add_ig_score_to_jsonl():
    # 读取langchain.jsonl文件
    if args.language == 'cn':
        langchain_file = os.path.join(args.test_path, f'{args.repo_name}.jsonl')
    elif args.language == 'en':
        langchain_file = os.path.join(args.test_path, f'{args.repo_name}_translation.jsonl')
    with open(langchain_file, 'r', encoding='utf-8') as f:
        langchain_data = [json.loads(line) for line in f]

    # 读取IG_scores.json文件
    ig_scores_file = os.path.join(args.test_path, 'IG_scores.json')
    with open(ig_scores_file, 'r') as f:
        ig_scores_data = json.load(f)

    # 遍历langchain_data中的每个元素
    for item in langchain_data:
        test_id = item['id']
        # 检查IG_scores_data中是否有对应的test_id
        if test_id in ig_scores_data:
            # 获取所有模型的IG_base值
            ig_base_scores = {}
            for model in consider_models:
                if model in ig_scores_data[test_id]:
                    ig_base = ig_scores_data[test_id][model].get('IG_base', 0)
                    ig_base_scores[model] = ig_base
                else:
                    ig_base_scores[model] = 0  # 如果模型不存在，使用0作为默认值
            # 添加到item中
            item['IG_base'] = ig_base_scores

    # 将更新后的数据写回到langchain.jsonl文件
    with open(langchain_file, 'w', encoding='utf-8') as f:
        for item in langchain_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Added IG_score to {len(langchain_data)} items in {langchain_file}")
'''
def count_positive_ig_base():
    # 读取langchain.jsonl文件
    langchain_file = os.path.join(args.test_path, f'{args.repo_name}.jsonl')
    with open(langchain_file, 'r', encoding='utf-8') as f:
        langchain_data = [json.loads(line) for line in f]

    # 初始化计数器
    positive_count = 0

    # 遍历每个元素，统计IG_base大于0的数量
    for item in langchain_data:
        if 'IG_score' in item and 'IG_base' in item['IG_score']:
            ig_base = item['IG_score']['IG_base']
            if ig_base > 0:
                positive_count += 1

    print(f"Number of IG_base values greater than 0: {positive_count}")
    return positive_count
'''
def calc_model_score():
    # 1. 从 test_results.json 读取结果
    results_file = os.path.join(args.test_path, 'test_results.json')
    with open(results_file, 'r') as f:
        result_data = json.load(f)  # 读取整个 JSON 文件为一个字典

    valid_ids = []
    testcases = {}
    file_path = os.path.join(args.test_path, f'{args.repo_name}.jsonl')
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if 'id' in json_obj:
                valid_ids.append(json_obj['id'])
                testcases[json_obj['id']] = json_obj
    
    # 初始化存储成功率的字典
    model_scores = {model: {'fill_success_rate': 0, 'fill_pass@all': 0,'empty_success_rate': 0, 'empty_pass@all': 0, 'valid_count': 0} for model in calc_models}

    # 2. 遍历每个测试点
    for test_id, model_results in result_data.items():
        # 检查该测试点是否是有效的
        is_valid = all(model in model_results and 'fill_result' in model_results[model] and 'empty_result' in model_results[model] for model in calc_models)
        if not test_id in valid_ids:
            continue
        if is_valid:
            # 计算每个模型的成功率
            total_passed = testcases[test_id]['pytest_info']['total_num']
            base_passed_num =  testcases[test_id]['pytest_info']['base_passed_num']
            
            
            for model in calc_models:
                fill_passed, fill_skipped, fill_failed = model_results[model]['fill_result']
                empty_passed, empty_skipped, empty_failed = model_results[model]['empty_result']

                # 计算通过率
                
                fill_success_rate = max((fill_passed - base_passed_num) / (total_passed - base_passed_num),0)
                print(testcases[test_id]["origin_file"])
                print(fill_success_rate)
                empty_success_rate = max((empty_passed- base_passed_num) / (total_passed - base_passed_num), 0) 

                fill_ac_all = 1 if fill_passed >= total_passed else 0
                empty_ac_all = 1 if empty_passed >= total_passed else 0
                # 累加成功率
                model_scores[model]['fill_success_rate'] += fill_success_rate
                model_scores[model]['empty_success_rate'] += empty_success_rate
                model_scores[model]['fill_pass@all'] += fill_ac_all
                model_scores[model]['empty_pass@all'] += empty_ac_all
                model_scores[model]['valid_count'] += 1

        else:
            print(f"[Error] Invalid test case: {test_id}")
        
    
    # 计算平均成功率
    for model in calc_models:
        if model_scores[model]['valid_count'] > 0:
            model_scores[model]['fill_success_rate'] /= model_scores[model]['valid_count']
            model_scores[model]['empty_success_rate'] /= model_scores[model]['valid_count']
            model_scores[model]['fill_pass@all'] /= model_scores[model]['valid_count']
            model_scores[model]['empty_pass@all'] /= model_scores[model]['valid_count']
    return model_scores
    


if __name__ == '__main__':
    # 1. 找到所有目标文件夹
    folder_paths = {}
    for root, dirs, files in os.walk(args.test_path):
        for folder in target_models:
            if folder in dirs:
                folder_path = os.path.join(root, folder)
                folder_paths.setdefault(folder, []).append(folder_path)
    
    # 把results转换JSON
    generate_result(folder_paths)
    #if not os.path.exists(os.path.join(args.test_path, f'{args.repo_name}.jsonl')) and not args.regenerate:
    write_pytest_info()
    calc_IG_score()
        #根据IG score表格，筛选出合格的测试点
    filter_testcases()

    #add_ig_score_to_jsonl()
        #count_positive_ig_base()



    # 把JSON的分数计算成表
    model_scores = calc_model_score()
    print('MODEL \t Scores ')
    for model, scores in model_scores.items():
        print(model, '\t', scores)

  


    
