import subprocess
import re
import ast
import textwrap  # 用于移除额外的缩进
import json
import os
import argparse
from variable_tracker import extract_lvalues_and_rvalues, extract_lvalues_new
import utils
import argparse
import traceback


parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='transformers', help='Repository name')
parser.add_argument('--regenerate', action='store_true', help='Regenerate the test cases')
parser.add_argument('--output_dir', type=str, default='/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/', help='Output directory for results')

args = parser.parse_args()

repo_name = args.repo_name
output_dir = args.output_dir
repo_args = utils.get_repo_args(args.repo_name)
mapping_path = repo_args["test_mapping_path"]
find_path = repo_args["find_path"]
copy_path = repo_args["copy_path"]
repo_path = repo_args["repo_path"]
repo_running_path = repo_args['repo_running_path']

directory, _ = os.path.split(mapping_path)

print(f"generating repo {repo_name}, \nmapping_path {mapping_path} \n")
print("-" * 40)
print("-" * 40)


ids = set()
with open(f"/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/fulingyue/AutoCoderBench/results/{repo_name}/{repo_name}.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        # 每行是一个 JSON 对象，读取并解析
        data = json.loads(line)
        # 获取 'id' 并添加到集合中
        id_value = data.get('id')
        if id_value is not None:
            ids.add(id_value)
    
print(ids)

def find_function_code_ast(file_path, target):
    # 提取类名和函数名
    if "::" in target:
        class_name, function_name = target.split("::")
    else:
        class_name, function_name = None, target

    with open(file_path, 'r') as file:
        code = file.read()

        
    # 解析代码为AST
    tree = ast.parse(code)
    if class_name is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                function_start_line = node.lineno
                function_end_line = node.end_lineno if hasattr(node, 'end_lineno') else None
                # 提取函数代码
                code_lines = code.splitlines()
                function_code = "\n".join(code_lines[function_start_line-1:function_end_line])
                return (1, len(code.splitlines())), (function_start_line, function_end_line), function_code
    else:
        # print(f'''find {class_name}::{function_name}''')
        # 遍历AST，寻找目标类和函数
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # 获取类的定义行范围
                class_start_line = node.lineno
                class_end_line = max(
                    (child.end_lineno if hasattr(child, 'end_lineno') else child.lineno)
                    for child in node.body
                ) if hasattr(node, 'body') and node.body else node.lineno

                # 提取类代码
                code_lines = code.splitlines()
                class_code = "\n".join(code_lines[class_start_line-1:class_end_line])

                # 遍历类的子节点，寻找函数定义
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == function_name:
                        # 获取函数的定义行范围
                        function_start_line = child.lineno
                        function_end_line = child.end_lineno if hasattr(child, 'end_lineno') else None
                        # 提取函数代码
                        function_code = "\n".join(code_lines[function_start_line-1:function_end_line])

                        return (class_start_line, class_end_line), (function_start_line, function_end_line), function_code

            # 如果找不到函数，返回整个类的起止行号，但函数信息为空
                return (class_start_line, class_end_line), None, f"未找到类 {class_name} 中的函数 {function_name}"

    # 如果找不到类，返回错误信息
        return None, None, f"未找到 {target}"

def count_condition(node, depth):
    if depth == 4:
        return 0
    name = node["name"]
    path = node["source_dir"]
    func_file = ""
    prune_count = 0
    # 递归处理孩子
    children = node["children"]
    for child in children:
        prune_count += count_condition(child, depth+1)
    # 检查是否是已有的题目
    if path != None:
        file_name = path.split('/')[-1].replace('.py', '')
        src_transformers_index = path.find(find_path)
        file_path = path[src_transformers_index + len(find_path):path.rfind('/')]
        if name.split(".")[-2] == path.split("/")[-1].split(".")[0]:
            func_file = name.split(".")[-1]+".py"
        elif name.split(".")[-3] == path.split("/")[-1].split(".")[0]:
            func_file = name.split(".")[-2] + "::" + name.split(".")[-1] + ".py"
        else:
            print(f"failed to extract for name {name}, path {path}")
        id = os.path.join(repo_name, file_path, file_name, func_file).replace(".py", "").replace("/", ".")
        if id in ids:
            prune_count += 1
    return prune_count
    

def func_problem(node, testcase_info, depth, count):
    if depth == 4:
        return 0
    name = node["name"]
    path = node["source_dir"]
    func_file = ""
    prune = False
    prune_count = 0 # 我和我之下的id的个数
    # 递归处理孩子
    children = node["children"]
    id = None
    temp_dict = {}
    if path and "__init__" in path:
        return 0

    # 检查是否是已有的题目
    if path != None:
        file_name = path.split('/')[-1].replace('.py', '')
        src_transformers_index = path.find(find_path)
        file_path = path[src_transformers_index + len(find_path):path.rfind('/')]

        if name.split(".")[-2] == path.split("/")[-1].split(".")[0]:
            func_file = name.split(".")[-1]+".py"
        elif len(name.split(".")) >= 3 and name.split(".")[-3] == path.split("/")[-1].split(".")[0]:
            func_file = name.split(".")[-2] + "::" + name.split(".")[-1] + ".py"
        else:
            print(f"failed to extract for name {name}, path {path}")
        # 找到行号
        source_code_path = os.path.join(repo_path, file_path, f'{file_name}.py')
        temp = find_function_code_ast(source_code_path, func_file.replace(".py", ""))
        # if temp is None:
        #     continue
        
        if temp is not None :
            class_lineno, func_lineno, func_code = temp
            #     continue
            
            temp_dict = {
                'class_start_lineno': class_lineno[0] if class_lineno is not None else None,
                'class_end_lineno': class_lineno[1] if class_lineno is not None else None,
                'func_start_lineno': func_lineno[0] if func_lineno is not None else None,
                'func_end_lineno': func_lineno[1] if func_lineno is not None else None, 
                'func_code': func_code,
            }
        
        id = os.path.join(repo_name, file_path, file_name, func_file).replace(".py", "").replace("/", ".")
        # print(id)
        if id in ids:
            prune = True
    if prune == True:
        for child in children:
            prune_count = max(prune_count, func_problem(child, testcase_info, depth+1, count+1))
        prune_count += 1
    else: 
        for child in children:
            prune_count = max(prune_count, func_problem(child, testcase_info, depth+1, count))
    if prune_count == 0:
        return prune_count
    if prune_count + count > 1 and path != None:
        # print(prune_count, count)
    # if True:
        print(prune_count, count)
        if id in ids and id not in set(testcase_info["id"]):
            testcase_info["id"].append(os.path.join(repo_name, file_path, file_name, func_file).replace(".py", "").replace("/", "."))
            
        if name not in set(testcase_info["node"]):
            testcase_info["origin_file"].append(path)
            testcase_info["prob_info"].append(temp_dict)
            testcase_info["node"].append(name)
            testcase_info["test"].append(f"prune_count{prune_count} count{count}")
            

    return prune_count

test_case_root_dir = os.path.join(output_dir, 'func_testcases', repo_name)
# func_testcase_mapping_path = os.path.join(test_case_root_dir, "testcase_mapping.jsonl")
# if os.path.exists(func_testcase_mapping_path):
#     os.remove(func_testcase_mapping_path)


with open(mapping_path, 'r', encoding='utf-8') as file:
    testcase_infos = []
    testcase_test_file = []
    for line_num, line in enumerate(file):
        data = json.loads(line.strip())
        test_file = data.get("test_file", "")
        origin_file = data.get("origin_file", "")
        test_path = test_file
        if test_path in testcase_test_file:
            continue
        file_name = origin_file.split('/')[-1].replace('.py', '')
        
        src_transformers_index = origin_file.find(find_path)
        file_path = origin_file[src_transformers_index + len(find_path):origin_file.rfind('/')]
        test_case_dir = os.path.join(output_dir, 'func_testcases', repo_name, test_path.replace(copy_path, "").replace(".py", ""))
        tree_path = os.path.join(test_case_dir, 'funcCallTree_new.json' )
        question_path = os.path.join(output_dir, 'test_case_info.jsonl')
        if not os.path.exists(test_case_dir):
            os.makedirs(test_case_dir)

        # 打印或使用提取的信息
        print(f"Test Path: {test_path}")
        # print(f"Origin File: {origin_file}")
        if not os.path.exists(tree_path) or args.regenerate:
        # #     with open (os.path.join(test_case_root_dir, "log.txt"), "a") as log_file:
        # #         log_file.write(f"{test_path} doesnt have extracted tree\n")
        # #     continue
        # if True:
            from function_tracker import track_function            
            try:
                print("track_function")
                return_code = track_function(repo_name, test_path, output_dir)
                if return_code == False:
                    with open (os.path.join(test_case_root_dir, "log.txt"), "a") as log_file:
                        log_file.write(f"{test_path} encountered error in function tracker\n")
                        print("{test_path} encountered error in function tracker\n")
            except Exception as e:
                log_dir = test_case_dir.replace("func_testcases", "func_testcases/log")
                print(log_dir)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                
                command = [
                    'python', 'function_tracker.py',
                    '--test_path', test_path,
                    '--repo_name', repo_name
                ]
                with open (os.path.join(log_dir, "function_tracker.txt"), "w") as output_file:
                    traceback.print_exc(file=output_file)
                    output_file.write(f"An exception occurred: {e.__class__.__name__}: {e}\n")
                    output_file.write("command: \n" + ' '.join(command))
                traceback.print_exc()                
                
        with open(tree_path, "r") as tree_file:
            data = json.load(tree_file)
            
            testcase_info = {
                "id": [],
                "project": repo_name,
                "origin_file": [],
                "test_list":[test_path],
                "prob_info": [],
                "type": ["Development","cross_file"],
                "node": [],
                "test": [],
                "language": "Python"
            }
            testcase_test_file.append(test_path)
            func_problem(data, testcase_info, 0, 0)
            with open (os.path.join(test_case_dir, 'func_testcases_info.json' ), "w") as save_file:
                json.dump(testcase_info, save_file, indent=4)
            # count = count_condition(data, 0)
            flag = False
            if len(testcase_info["id"]) > 1:
                flag = True
                for pos in testcase_info["prob_info"]:
                    if pos == {}:
                        flag = False
            if flag:            
                test_log_dir = os.path.join(test_case_dir.replace("func_testcases", "func_testcases/log"), "log_test.txt")
                if os.path.exists(test_log_dir):
                    from utils import read_log
                    passed, skipped, failed = read_log(test_log_dir)
                else:
                    passed, skipped, failed = [0,0,1]
                testcase_info["pytest_info"] = [passed, skipped, failed]
                testcase_infos.append(testcase_info)
                
                with open (os.path.join(test_case_root_dir, "log.txt"), "a") as log_file:
                    log_file.write(f"{test_path} is a generated problem \n")
        print("-" * 40)
    
    with open (os.path.join(test_case_root_dir, 'func_testcases_info_new.json' ), "w") as save_file:
        json.dump(testcase_infos, save_file, indent=4)
    print(len(testcase_infos))

        # utils.get_response("hello")
        
