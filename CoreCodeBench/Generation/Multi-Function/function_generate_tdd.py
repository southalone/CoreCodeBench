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
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='transformers', help='Repository name')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
parser.add_argument('--regenerate', action='store_true', help='Regenerate the test cases')
parser.add_argument('--retrack', action='store_true', help='Retrack the test files')
args = parser.parse_args()
print(args.regenerate)
args.regenerate = True
# args.retrack = True
repo_name = args.repo_name
output_dir = args.output_dir
repo_args = utils.get_repo_args(args.repo_name)
mapping_path = repo_args["test_mapping_path"]
find_path = repo_args["find_path"]
copy_path = repo_args["copy_path"]
repo_path = repo_args["repo_path"]
repo_running_path = repo_args['repo_running_path']

running_path_copy = repo_running_path.replace(repo_path, copy_path)



directory, _ = os.path.split(mapping_path)

print(f"generating repo {repo_name}, \nmapping_path {mapping_path} \n")
print("-" * 40)
print("-" * 40)


ids = set()
gen_ids = set()
with open(f"./TDD_results/{args.repo_name}/{args.repo_name}.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        # 每行是一个 JSON 对象，读取并解析
        data = json.loads(line)
        # 获取 'id' 并添加到集合中
        id_value = data.get('id')
        if id_value is not None:
            ids.add(id_value)
            gen_ids.add(id_value)



if os.path.exists(f"./func_testcases/{repo_name}/{repo_name}_tools.jsonl"):
    with open (f"./func_testcases/{repo_name}/{repo_name}_tools.jsonl", "r") as f:
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

    
def check_comment(func_code):
    # Check whether the docstring is enclosed with single quotes (`\'\'\'`) or double quotes (`\"\"\"`).
    # - Docstring Quote Type: <1 if single quotes, 2 if double quotes, and 0 if no docstring>

    prompt = f"""
Given the source code of a function, determine if there is a comment at the beginning explaining its purpose. The comment should be in the form of a docstring and contains input and output description. Return the docstring type and function definition and docstring without function implementation using the following format, you should keep the original indentation:
- Docstring Quote Type: <1 if docstringed, and 0 if no docstring>
<code>
<extracted code>
</code>


Example:


For the following function:
<code>
def to_string(self):
    \"\"\"
    Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
    version of the image.
    \"\"\"
    # Function implementation
</code>
The expected output is:
Docstring Quote Type: 0


For the following function:
<code>
def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        \"\"\"
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ALBERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        \"\"\"
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep
</code>
The expected output is:
Docstring Quote Type: 1
<code>
def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        \"\"\"
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ALBERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        \"\"\"
</code>

For the following function:
<code>
    def add(self, word: str):
        \"\"\"
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` in `self._termination_char` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged
        \"\"\"
        if not word:
            # Prevent empty string
            return

        self._tokens.add(word)
        ref = self.data
        ref[self._termination_char] = 1    
</code>
The expected output is:
Docstring Quote Type: 1
<code>
    def add(self, word: str):
        \"\"\"
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` in `self._termination_char` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged
        \"\"\"
</code>

check the following function:
<code>
{func_code}
</code>
"""
    response = utils.get_response(prompt, "gpt4o")
    # 提取结果
    docstring_match = re.search(r'Docstring Quote Type:\s*(?P<quote_type>\d)', response)
    docstring_type = docstring_match.group('quote_type') if docstring_match else None
    
    if not docstring_match:
        print("Failed to extract type")
        print(response)
        print("!"*40)
        return None
    
    if docstring_type != "0":
        code_match = re.search(r'<code>\s*(?P<extracted_code>[\s\S]*?)</code>', response)
        new_func_code = code_match.group('extracted_code') if code_match else None
        if not new_func_code:
            print(response)
            print("!!!!failed to extract code!!!!")
            return None
        new_func_code = new_func_code.rstrip()
        if new_func_code in func_code:
            return new_func_code
        else:
            print(response)
            print("~"*40)
            print(new_func_code)
            print("~"*40)
            print(func_code)
            # 1/0
            print("not in func_code")
            print("!"*40)
            return None
        # print(response)
        # print(func_code)
    return None

def func_problem(node, testcase_info, depth, count):
    if depth == 4:
        return 0
    if node == None:
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
    
    if testcase_info["func_count"] > 3:
        return 0
    
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
        id = os.path.join(repo_name, file_path, file_name, func_file).replace(".py", "").replace("/", ".")
        # print(id)
        if id in ids:
            prune = True
            
        
        if temp is None:
            return prune_count
            
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
            
            if func_lineno is None:
                temp_dict = {}
        
            if not prune and not "__init__" in name and testcase_info["toolfunc_count"] < 3  and func_lineno and func_lineno[1] - func_lineno[0] > 10:
                new_func_code = check_comment(func_code)
                # 检查是否是希望的工具函数
                if new_func_code:
                    
                    new_func_code += "\n <complete code here>"
                    print(new_func_code)
                    test_path = testcase_info["test_list"][0]
                    created_testcase = {
                        "id": id,
                        "project": repo_name,
                        "func": func_file.replace(".py", ""),
                        "origin_file": os.path.relpath(path, running_path_copy),
                        "test_list" : os.path.relpath(test_path, running_path_copy),
                        "prob_info":{
                            'class_start_lineno': class_lineno[0] if class_lineno is not None else None,
                            'class_end_lineno': class_lineno[1] if class_lineno is not None else None,
                            'func_start_lineno': func_lineno[0] if func_lineno is not None else None,
                            'func_end_lineno': func_lineno[1] if func_lineno is not None else None, 
                            'key_block_start_lineno': (func_lineno[0] + len(new_func_code.splitlines()) - 1) if func_lineno is not None else None,
                            'key_block_end_lineno': func_lineno[1] if func_lineno is not None else None,
                            'new_func_code': new_func_code,
                        }
                    }
                    json_line = json.dumps(created_testcase,ensure_ascii=False)
                    ids.add(id)
                    with open (f"./func_testcases/{repo_name}/{repo_name}_tools.jsonl", "a") as new_testcase_file:
                        new_testcase_file.write(json_line + "\n")
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
        print(prune_count, count)
        if id in ids and id not in set(testcase_info["id"]) and testcase_info["func_count"] < 7:
            if id not in gen_ids:
                testcase_info["toolfunc_count"] += 1
            testcase_info["func_count"] += 1
            
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
    for line_num, line in tqdm(enumerate(file)):
        
        origin_data = json.loads(line.strip())
        test_file = origin_data.get("test_file", "")
        origin_file = origin_data.get("origin_file", "")
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
        if not os.path.exists(tree_path) or args.retrack:
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
        if not os.path.exists(tree_path):
            continue              
                
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
                "language": "Python",
                "toolfunc_count": 0,
                "func_count":0
            }
            testcase_test_file.append(test_path)
            if os.path.exists(os.path.join(test_case_dir, 'func_tdd_testcase_info.json' )) and not args.regenerate:
                with open (os.path.join(test_case_dir, 'func_tdd_testcase_info.json' ), "r") as save_file:
                    testcase_info = json.load(save_file)
                if len(testcase_info["id"]) > 1:
                
                    if testcase_info["toolfunc_count"] == 0:
                        testcase_info["type"] = ["development"]
                    elif testcase_info["toolfunc_count"] == len(testcase_info["id"]):
                        testcase_info["type"] = ["function_empty"]
                    else:
                        testcase_info["type"] = ["function_empty", "development"]
                testcase_info["pytest_info"] = {"total_num": origin_data["pytest"]["passed"]}
            else:
                func_problem(data, testcase_info, 0, 0)
                if len(testcase_info["id"]) > 1:
                    if testcase_info["toolfunc_count"] == 0:
                        testcase_info["type"] = ["development"]
                    elif testcase_info["toolfunc_count"] == len(testcase_info["id"]):
                        testcase_info["type"] = ["function_empty"]
                    else:
                        testcase_info["type"] = ["function_empty", "development"]
            with open (os.path.join(test_case_dir, 'func_tdd_testcase_info.json' ), "w") as save_file:
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
                # if os.path.exists(test_log_dir):
                #     from utils import read_log
                #     passed, skipped, failed = read_log(test_log_dir)
                # else:
                #     passed, skipped, failed = [0,0,1]
                testcase_info["pytest_info"] = {"total_num": origin_data["pytest"]["passed"]}
                testcase_infos.append(testcase_info)
                
                with open (os.path.join(test_case_root_dir, "log_tdd.txt"), "a") as log_file:
                    log_file.write(f"{test_path} is a generated problem \n")
        print("-" * 40)
        print(len(testcase_infos))
    
    # with open (os.path.join(test_case_root_dir, 'func_testcases_tools_info.json' ), "w") as save_file:
    #     json.dump(testcase_infos, save_file, indent=4)
    with open (os.path.join(test_case_root_dir, 'func_tdd_testcases_info.jsonl' ), "w") as save_file:
        for testcase_info in testcase_infos:
            json_line = json.dumps(testcase_info, ensure_ascii=False)
            save_file.write(json_line + "\n")
    
with open (f"./func_testcases/{repo_name}/{repo_name}_tdd.jsonl", "w") as save_file:
    with open(f"./TDD_results/{args.repo_name}/{args.repo_name}.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            save_file.write(line)
    if os.path.exists(f"./func_testcases/{repo_name}/{repo_name}_tools.jsonl"):
        with open (f"./func_testcases/{repo_name}/{repo_name}_tools.jsonl", "r") as f:
            for line in f:
                save_file.write(line)

