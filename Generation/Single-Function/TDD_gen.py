import json
import ast
import textwrap
import re
import os
from variable_tracker import extract_lvalues_and_rvalues
import utils
import argparse
import shutil
import subprocess

LINE_NUM_LIMIT = 50
LINE_NUM_MIN = 4

parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='', help='name of repo')
args = parser.parse_args()
repo_name = args.repo_name

repo_data = repo_info[repo_name]
copy_path = repo_data.get('copy_path', '')
repo_path = repo_data.get('repo_path', '')
test_path_relative = repo_data.get('_test_path', '').lstrip('/')
src_path_relative = repo_data.get('_src_path', '').lstrip('/')
src_path = os.path.join(copy_path, src_path_relative)
repo_args = utils.get_repo_args(repo_name)
mapping_path = repo_args["test_mapping_path"]
find_path = repo_args["find_path"]
relative_running_path = repo_args["relative_running_path"]
repo_running_path = repo_args["repo_running_path"]
copy_running_path = repo_args["copy_running_path"]
copy_root_path = repo_args["copy_root_path"]
root_path = repo_args["root_path"]
output_dir = repo_args["root_path"]


def extract_lvalues_new(key_block_start, key_block_end, code):
    file_name = code['file_name']
    file_path = code['file_path']
    test_case_dir = code['test_case_dir']
    
    copy_code_path = os.path.join(copy_path, file_path, f'{file_name}.py')
    test_path = code['test_file']
    print(f"test_path:{test_path}")
    copy_test_path = test_path.replace(repo_path, copy_path)

    print(key_block_start, key_block_end)
    print(test_case_dir)
    print(copy_code_path)
    temp_file = os.path.join(test_case_dir, 'lhs.tmp')
    # 把copy_test_path中的文件复制一份
    copy_copy_test_path = copy_test_path.replace('.py', '_copy.py')
    print(f"copy_test_path:{copy_test_path}")
    print(f"copy_copy_test_path:{copy_copy_test_path}")
    shutil.copyfile(copy_test_path, copy_copy_test_path)

    with open(copy_copy_test_path, 'r') as f:
        code = f.read()
    
    TIMEOUT=120
    if 'import unittest' in code:
        # 用unittest进行变量追踪
        prefix_code= f'''import sys
import torch
import copy
from deepdiff import DeepDiff

template_file = '{temp_file}'
class VariableTracker:
    def __init__(self, start_line, end_line, target_file):
        self.start_line = start_line
        self.end_line = end_line
        self.target_file = target_file
        self.previous_locals = {{}}
        
        with open(template_file, 'w') as f:
            f.write('')

    def safe_compare(self, var_name, current_value, previous_value):
        """Safely compare two values, returning True if they differ."""
        try:
            # 首先检查内存地址
            if id(current_value) == id(previous_value):
                return False
            
            # 使用 DeepDiff 进行深度比较
            diff = DeepDiff(current_value, previous_value, ignore_order=True)
            
            # 如果有差异，返回 True
            return bool(diff)
        except Exception as e:
            print(f"Comparison failed for variable '{{var_name}}': {{e}}")
            return True

    def trace_func(self, frame, event, arg):
        if (event == "line" and
            frame.f_code.co_filename.endswith(self.target_file) and
            self.start_line <= frame.f_lineno <= self.end_line):
            
            current_locals = copy.deepcopy(frame.f_locals)

            # 初始化时记录所有变量
            if not self.previous_locals:
                self.previous_locals = current_locals

            # 检查哪些变量发生了变化
            changed_vars = {{
                var: current_locals[var]
                for var in current_locals
                if (var not in self.previous_locals or
                    self.safe_compare(var, current_locals[var], self.previous_locals[var]))
            }}

            # 更新之前的局部变量状态
            self.previous_locals = current_locals

            if changed_vars:
                print('line: ', frame.f_lineno, 'changed_vars: ', changed_vars)
                with open(template_file, 'a') as f:
                    f.write('\\n'.join(list(changed_vars.keys())))
                    f.write('\\n')

        return self.trace_func

    def track_variable_changes(self, func):
        def wrapper(*args, **kwargs):
            sys.settrace(self.trace_func)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
        return wrapper

# 设置行号范围
tracker = VariableTracker(start_line={key_block_start}, end_line={key_block_end}, target_file='{copy_code_path}')

# 设置全局跟踪函数
sys.settrace(tracker.trace_func)

'''
        suffix_code = '''
    if  __name__ == "__main__":
        unittest.main()
    # 取消跟踪
    sys.settrace(None)'''
        new_code = prefix_code + code + suffix_code
        with open(copy_copy_test_path, 'w') as f:
            f.write(new_code)    
        print('UNITESTING LHS....')
        env = os.environ.copy()
        env["PYTHONPATH"] = copy_running_path
        module_name = copy_copy_test_path.replace(copy_path,'').replace('.py','').replace('/','.')
        print(f"modulename:{module_name}")
        print(f"copy_path:{copy_copy_test_path}")
        print(f"copy_running_path:{copy_running_path}")
        print(f"cwd: {copy_path}")
        try:
            subprocess.run(['python', '-m', module_name], cwd=copy_path, env=env, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"Unittest {module_name} exceeded time limit and was terminated.")
        
    else:
        prefix_code= f'''import sys
import torch
import copy
from deepdiff import DeepDiff
import pytest
import os
def pytest_addoption(parser):
    parser.addoption(
        "--only-extended",
        action="store_true",
        default=False,
        help="Run only extended tests (dummy fix)"
    )
template_file = '{temp_file}'
class VariableTracker:
    def __init__(self, start_line, end_line, target_file):
        self.start_line = start_line
        self.end_line = end_line
        self.target_file = target_file
        self.previous_locals = {{}}
        
        with open(template_file, 'w') as f:
            f.write('')

    def safe_compare(self, var_name, current_value, previous_value):
        """Safely compare two values, returning True if they differ."""
        try:
            # 首先检查内存地址
            if id(current_value) == id(previous_value):
                return False
            
            # 使用 DeepDiff 进行深度比较
            diff = DeepDiff(current_value, previous_value, ignore_order=True)
            
            # 如果有差异，返回 True
            return bool(diff)
        except Exception as e:
            print(f"Comparison failed for variable '{{var_name}}': {{e}}")
            return True

    def trace_func(self, frame, event, arg):
        
        if (event == "line" and
             frame.f_code.co_filename.endswith(self.target_file)  and
            self.start_line <= frame.f_lineno <= self.end_line):
            
            current_locals = copy.deepcopy(frame.f_locals)

            # 初始化时记录所有变量
            if not self.previous_locals:
                self.previous_locals = current_locals

            # 检查哪些变量发生了变化
            changed_vars = {{
                var: current_locals[var]
                for var in current_locals
                if (var not in self.previous_locals or
                    self.safe_compare(var, current_locals[var], self.previous_locals[var]))
            }}

            # 更新之前的局部变量状态
            self.previous_locals = current_locals

            if changed_vars:
                print('line: ', frame.f_lineno, 'changed_vars: ', changed_vars)
                with open(template_file, 'a') as f:
                    f.write('\\n'.join(list(changed_vars.keys())))
                    f.write('\\n')

        return self.trace_func

    
@pytest.fixture(scope="module", autouse=True)
def tracker():
    # Initialize the tracker
    tracker = VariableTracker(start_line={key_block_start}, end_line={key_block_end}, target_file='{copy_code_path}')
    sys.settrace(tracker.trace_func)
    yield tracker
    # Cleanup the tracker
    sys.settrace(None)
'''

        suffix_code = '''
if __name__ == "__main__":
    pytest.main([__file__])'''
        
        new_code = prefix_code + code + suffix_code
        with open(copy_copy_test_path, 'w') as f:
            f.write(new_code)   
        print('PYTESTING LHS....') 
        
        env = os.environ.copy()
        env["PYTHONPATH"] = copy_running_path
        module_name = copy_copy_test_path.replace(copy_path,'').replace('.py','').replace('/','.')
        print(f"modulename:{module_name}")
        print(f"copy_path:{copy_copy_test_path}")
        print(f"copy_running_path:{copy_running_path}")
        print(f"cwd: {copy_path}")
        try:
            subprocess.run(['python', '-m', module_name], cwd=copy_path, env=env, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"Unittest {module_name} exceeded time limit and was terminated.")
        
    try:       
        with open(temp_file, 'r') as f:
            lvalues = f.read().split('\n')
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        os.remove(copy_copy_test_path)
        return None
    # 把lhs.tmp删除
    os.remove(temp_file)
    # 把copycopy_test_path删除
    os.remove(copy_copy_test_path)

    return lvalues

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

def generate_code_blocks_dict(func_name, code):
    """
    分析 Python 代码块并生成字典，key 为 '#startlineno#endlineno' 的格式，
    value 为包含行号、节点类型和代码片段的字典。
    """
    # 解析代码为 AST
    
    code = textwrap.dedent(code)
    tree = ast.parse(code)

    # 结果字典
    blocks_dict = {}

    def analyze_block(node, code_lines):
        """
        递归分析一个节点，生成字典项。
        """
        start_line = getattr(node, 'lineno', None)  # 起始行号
        end_line = getattr(node, 'end_lineno', None)  # 结束行号
        block_type = type(node).__name__  # 节点类型
        # print(block_type, start_line, end_line)
        # 函数头的部分不保存
        if block_type == 'arg':
            return
        # 如果节点有行号信息，提取代码片段
        
        if start_line and end_line:
            block_code = "\n".join(code_lines[start_line - 1:end_line])
            # 构造 key
            key = f"{func_name}#{start_line}#{end_line}"
            # 存储 block 信息
            blocks_dict[key] = {
                "block_type": block_type,
                "start_line": start_line,
                "end_line": end_line,
                "code": block_code,
            }
        # 递归分析子节点
        
        for child in ast.iter_child_nodes(node):
            analyze_block(child, code_lines)

    # 分析整棵 AST 树
    code_lines = code.splitlines()
    analyze_block(tree, code_lines)
    return blocks_dict

def find_key_block(func, code, recur = None):
    if recur:
        chat_message = f'''关键代码块定义为：实现函数主要功能的代码部分，直接决定函数是否能完成预期目标；执行效率显著影响函数性能的代码部分。
请你根据函数{func}的代码，找到block {recur}中的关键代码块，输出其子关键代码块的block_id。要求选出来的代码块总行数不超过60行，所以请谨慎选择，确保选择的是最为重要的部分。
输出格式：
你可以选择多个**连续**的block，这时请输出block_id的列表：
```python 
blocks = ["blockid1", "blockid2", ...]
```
如果实现的函数比较简单，仅仅包含初始化或者返回值，则说明该函数不存在关键代码块，这时请输出
```python 
blocks = None
```
请不要在代码段中输出额外的注释说明，只输出blockid。
    请在{recur}代码块的子代码块中选择关键代码块。
    函数代码：
    {code['func_code']}
    函数block信息：
    {code['block_info']}
    '''
    else:
        chat_message = f'''关键代码块定义为：实现函数主要功能的代码部分，直接决定函数是否能完成预期目标；执行效率显著影响函数性能的代码部分。
请你根据函数{func}的代码，找到其实现过程中的关键代码块，输出关键block的block_id。要求选出来的代码块总行数不超过60行，所以请谨慎选择，确保选择的是最为重要的部分。
输出格式：
你可以选择多个**连续**的block，这时请输出block_id的列表：
```python 
blocks = ["blockid1", "blockid2", ...]
```
如果实现的函数比较简单，仅仅包含初始化或者返回值，则说明该函数不存在关键代码块，这时请输出
```python 
blocks = None
```
请不要在代码段中输出额外的注释说明，只输出blockid。
函数代码：
{code['func_code']}
函数block信息：
{code['block_info']}
    '''
    chosen_block = utils.get_response(chat_message, model='gpt4o')
    if 'None' in  chosen_block:
       return 1e9, 0, None

    match = re.search(r'blocks\s*=\s*\[(.*?)\]', chosen_block, re.S)
    if match:
        block_content= match.group(1).strip()
    else:
        # print(chosen_block)
        print(f'函数{func}选择失败')
        return 1e9, 0, None

    blocks = block_content.split(',')
    block_list = []
    for block in blocks:
        block_list.append(block.strip().strip('"').strip())
    
    
    start_line_no = 1e9
    end_line_no = 0
    if block_list is None:
        return 1e9, 0, None
    
    for block in block_list:
        try:
            start_line, endline = code['block_info'][block]['start_line'], code['block_info'][block]['end_line']
            if max(endline, end_line_no) - min(start_line, start_line_no) > LINE_NUM_LIMIT:
                continue
            start_line_no = min(start_line, start_line_no)
            end_line_no = max(endline, end_line_no)
                 
        except Exception as e:
            print('ERROR', e)
            continue
    
    if start_line_no == 1e9 or end_line_no == 0:
        # 说明没有找到合适的block
        return find_key_block(func, code, recur = block_list)
    
    start_line_no = 2 if start_line_no == 1 else start_line_no # 保证函数头至少保留一行
    # print(f'''函数{func}的keyblocks为{blocks}，筛选后的行号为[{start_line_no},{end_line_no}]''')
    return start_line_no, end_line_no, block_list

def generate_new_code(func, code, class_code, class_start_lineno, start_line_no, end_line_no, model, score_model='claude3.5'):
    lines = code['func_code'].splitlines()
    import_code = code['import_code']
    # print('FUNC:', func)
    if start_line_no < 1 or end_line_no > len(lines) or start_line_no > end_line_no:
        raise ValueError("Invalid line range specified.")
    res = lines[:start_line_no-1] + ['<complete code here>']+ lines[end_line_no:]

#     key_block = '\n'.join(lines[start_line_no-1:end_line_no])
#     if not utils.is_core_code(key_block):
#         return None, None
#     lhs_var, rhs_var = extract_lvalues_and_rvalues(key_block, '\n'.join(lines[end_line_no:]))
#     lhs_var_2 = extract_lvalues_new(class_start_lineno+start_line_no-1, class_start_lineno+end_line_no-1, code)
#     lhs_var = list(set(lhs_var).union(set(lhs_var_2)))
#     # print(f'LHS:{lhs_var}, RHS: {rhs_var}')
#     if lhs_var is None or rhs_var is None:
#         return None, None
#     variable_list = list(set(lhs_var).intersection(set(rhs_var)))
#     # print(variable_list)
  


#     chat_message = f'''请结合上下文，分析给出的代码块，并以简洁的语言，按照给定的格式输出其功能（不要输出额外的内容）：
# 1. **目的**
#     描述代码块的主要目标和它在整个程序中的作用。特别是其在当前函数中的职责是什么。
# 2. **逻辑**
#     详细阐述代码块的核心逻辑和操作过程。对于所有的条件分支(if语句），需逐一解释。
#     如果涉及复杂的变量更新，请使用Markdown格式的公式来表示这些数学计算。
#     如果用到了代码块前文的变量，请尽量使用变量名来描述，并用反引号将变量名框出。用到的函数请用反引号将其框出，可以用```函数名(参数)```的形式，或者```函数名```的形式，请不要出现```函数名()```等会引起歧义的形式。
# 3. **异常**
#     如果待分析的代码块中抛出异常，请说明其抛出的异常情况及异常类型。如果代码块中无异常抛出，则此项写“无”。
# 4. **变量赋值**
#     根据给出的变量列表，用列表形式给出代码块中计算该变量的具体意义和作用。
#     如果表格中有识别错误的变量（例如后文没有用到），你可以直接删去该变量。如果变量列表中漏掉了某个被修改的变量（特别是`self.blockid_list.append(block)`这样的形式）请在列表中补充。
#     变量列表：{variable_list}

    
# ### 示例输出：
# 1. **目的**
#     解析目标字符串，提取其中的关键信息。目标字符串的格式为``` blocks = ["blockid1", "blockid2", ...]```，此代码块提取所有有效的blockid，生成一个新的字符串列表。
# 2. **逻辑**
#     使用正则表达式(re库)从目标字符串中提取blockid列表，随后遍历该列表，验证每个blockid是否在数据库中存在，并将其转换成整数类型后存入新列表。
# 3. **异常**
#     - `ValueError`： 如果目标字符串的格式不正确，无法提取有效的blockid列表，则抛出该异常。
# 4. **变量赋值**
#     - `self.blockid_list`：存储提取并验证后的blockid  

# ### 待分析的代码块:
# ```
# {key_block}
# ```
# ### 代码块的上下文信息：
# ```
# {class_code}
# ```
#     '''
#     explanation = utils.get_response(chat_message, model)
#     scores, response = utils.validate_code(explanation, key_block, model=score_model)
#     # print('第一轮得分：', scores)
#     # print('第一轮反馈：', response)
#     if sum(scores) == 0:
#         return None, None
#     if sum(scores) < 6:
#         modify_prompt = f'''
# 代码审查员认为生成的代码解释存在以下问题：
# ```
# {response}
# ```
# 请根据代码块的内容和审查员的建议，修改当前的代码解释，并按照规定格式输出，**不要输出额外的内容**。
# ### 待分析的代码块:
# ```
# {key_block}
# ```
# ### 当前的代码解释：
# {explanation}

# ### 输出要求：
# 1. **目的**
#     描述代码块的主要目标和它在整个程序中的作用。特别是其在当前函数中的职责是什么。
# 2. **逻辑**
#     详细阐述代码块的核心逻辑和操作过程。对于所有的条件分支(if语句），需逐一解释。
#     如果涉及复杂的变量更新，请使用Markdown格式的公式来表示这些数学计算。
#     如果用到了代码块前文的变量，请尽量使用变量名来描述，并用反引号将变量名框出。
# 3. **异常**
#     如果待分析的代码块中抛出异常（`raise`语句，不包括`except`语句），请说明其抛出的异常情况及异常类型。如果代码块中无异常抛出，则此项写“无”。
# 4. **变量赋值**
#     根据给出的变量列表，用列表形式给出代码块中计算该变量的具体意义和作用。
#     如果有识别错误的变量（例如后文没有用到），你可以直接删去该变量。如果变量列表中漏掉了某个被修改的变量（特别是`self.blockid_list.append(block)`这样的形式）请在列表中补充。


# ### 示例输出：
# 1. **目的**
#     解析目标字符串，提取其中的关键信息。目标字符串的格式为``` blocks = ["blockid1", "blockid2", ...]```，此代码块提取所有有效的blockid，生成一个新的字符串列表。
# 2. **逻辑**
#     使用正则表达式(re库)从目标字符串中提取blockid列表，随后遍历该列表，验证每个blockid是否在数据库中存在，并将其转换成整数类型后存入新列表。
# 3. **异常**
#     - `ValueError`： 如果目标字符串的格式不正确，无法提取有效的blockid列表，则抛出该异常。
# 4. **变量赋值**
#     - `self.blockid_list`：存储提取并验证后的blockid  

#     '''
#         new_explanation = utils.get_response(modify_prompt, model)
#         scores_new, _ = utils.validate_code(explanation, key_block)
#         if sum(scores_new) < 5 or 0 in scores_new:
#             # print(scores_new)
#             print(f"生成的代码解释仍然不符合要求，无法生成测试用例 ID{func}")
#             return None, None
#         explanation = new_explanation if sum(scores_new) > sum(scores) else explanation
#         scores = scores_new if sum(scores_new) > sum(scores) else scores

#     file_path = code['file_path']
#     file_name = code['file_name']
#     test_case_dir = os.path.join(output_dir, 'testcases', repo_name, file_path, file_name) 
#     # print('输出结果到：', test_case_dir)
#     with open(os.path.join(test_case_dir, f'{func}.py'), 'w') as file:
#         file.write(code['func_code'])

#     res = lines[:start_line_no-1] + ["# 本段代码的功能解释："] + ['#'+ e for e in explanation.splitlines()] + ['<complete code here>']+ lines[end_line_no:]
#     with open(os.path.join(test_case_dir, f'{func}_new.py'), 'w') as file:
#        file.write("\n".join(res))
    return "\n".join(res), scores

def get_import_code(whole_file):
    lines = whole_file.splitlines()
    import_code = []
    for line in lines:
        if line.startswith('import') or line.startswith('from'):
            import_code.append(line)
    return '\n'.join(import_code)


def extract_third_level_content(json_path, mapping_file_path, TDD_results_path):
    # print(f"Running extract_third_level_content...... {json_path}")
    json_file_path = json_path + '/funcCallTree_new.json'
    output_file_path = json_path + '/funcCallTree3level_valid.json'
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 读取映射文件，提取所有的origin_file路径
    origin_files = set()
    origin_files_name = set()
    with open(mapping_file_path, 'r') as mapping_file:
        for line in mapping_file:
            mapping = json.loads(line)
            origin_files.add(mapping['origin_file'])

            origin_file_name = mapping['origin_file']
            prefix = "AutoCoderBench/Source_Copy/"
            start_index = origin_file_name.find(prefix)
            if start_index != -1:
                origin_file_name = origin_file_name[start_index + len(prefix):]
            else:
                origin_file_name = None
            origin_files_name.add(origin_file_name)
        
    # 提取第三层内容
    third_level_content = []
    seen_names = set()
    for child in data.get('children', []):
        for sub_child in child.get('children', []):
            # print(f"find {sub_child.get('name')}")
            name = sub_child.get("name")
            source_dir = sub_child.get("source_dir")

            prefix = "AutoCoderBench/Source_Copy/"
            start_index = source_dir.find(prefix)
            if start_index != -1:
                source_dir_name = source_dir[start_index + len(prefix):]
            else:
                source_dir_name = None
            
            if name not in seen_names and source_dir_name in origin_files_name:
                print(name)
                seen_names.add(name)
                last_slash_index = source_dir.rfind('/')
                py_index = source_dir.find('.py')
                if last_slash_index != -1 and py_index != -1:
                    node_source = source_dir[last_slash_index + 1:py_index]

                start_index = name.find(node_source)
                remaining_part = name[start_index + len(node_source):]
                dot_count = remaining_part.count('.')
                if dot_count == 1:
                    func = remaining_part.split('.')[-1]
                elif dot_count > 1:
                    parts = remaining_part.rsplit('.', 2)
                    func = parts[-2] + '::' + parts[-1]


                temp = find_function_code_ast(source_dir, func)
                # print(temp)
                if temp is None:
                    continue
                class_lineno, func_lineno, func_code = temp
                if class_lineno  is None or func_lineno is None:
                    continue
                with open(source_dir, 'r') as file:
                    whole_file = file.read()
                import_code = get_import_code(whole_file)

                origin_file = source_dir
                test_file = None
                with open(mapping_file_path, 'r') as mapping_file:
                    for line in mapping_file:
                        mapping = json.loads(line.strip())

                        mapping_file_name = mapping['origin_file']
                        prefix = "AutoCoderBench/Source_Copy/"
                        start_index = mapping_file_name.find(prefix)
                        if start_index != -1:
                            mapping_file_name = mapping_file_name[start_index + len(prefix):]
                        else:
                            mapping_file_name = None
                        if mapping_file_name == source_dir_name:
                            test_file = mapping['test_file']
                            passed = mapping['pytest']['passed']
                            break
                
                test_path = test_file
                file_name = origin_file.split('/')[-1].replace('.py', '')
                src_transformers_index = origin_file.find(find_path)
                file_path = origin_file[src_transformers_index + len(find_path):origin_file.rfind('/')]
                test_case_dir = os.path.join(output_dir, 'testcases', repo_name, file_path, file_name) 

                code = {
                    "name": name,
                    "func_name": func,
                    "source_dir": sub_child.get("source_dir"),
                    "call_position": sub_child.get("call_position"),
                    'class_start_lineno': class_lineno[0],
                    'class_end_lineno': class_lineno[1],
                    'func_start_lineno': func_lineno[0],
                    'func_end_lineno': func_lineno[1],
                    'func_code': func_code,
                    'block_info': generate_code_blocks_dict(func, func_code),
                    'import_code': import_code,
                    "file_name": file_name,
                    "file_path":file_path,
                    "test_case_dir":test_case_dir,
                    "origin_file":origin_file,
                    "test_file":test_file
                }
                start_line_no, end_line_no, block_list= find_key_block(func, code)
                if start_line_no == 1e9 or end_line_no == 0 or start_line_no >= end_line_no or end_line_no - start_line_no < LINE_NUM_MIN:
                    continue
                # class_code = '\n'.join(whole_file.splitlines()[code['class_start_lineno']-1:code['class_end_lineno']])
                # new_code, scores = generate_new_code(func,code, class_code, code['func_start_lineno'], start_line_no, end_line_no, model='gpt4o', score_model='claude3.5')

                lines = code['func_code'].splitlines()
                prefix = lines[:start_line_no-1]
                suffix = lines[end_line_no:]
                placeholder = ['<complete code here>']
                new_code = '\n'.join(prefix + placeholder + suffix)

                # if new_code is None or scores is None:
                if new_code is None:
                    continue

                # LLM_Score = {"readability_score": scores[0], "accuracy_score": scores[1], "completeness_score": scores[2]}
                LLM_Score = {"readability_score": "", "accuracy_score": "", "completeness_score": ""}

                id_prefix = copy_running_path.replace(copy_root_path, "")
                id_prefix = id_prefix.replace("/", ".")
                id = id_prefix + name
                if "::" in func:
                    id = id[::-1].replace('.', '::', 1)[::-1]
                print(f"id: {id}, func: {func}")
                func_name = func

                from utils import read_log
                source_code_path = os.path.join(repo_path, file_path, f'{file_name}.py')
                result_dir = os.path.join(output_dir, 'retest', repo_name, file_path,  file_name)
                save_name = "retest"
                
                temp_dir = os.path.join(output_dir, 'tmp_source_code')
                os.makedirs(temp_dir, exist_ok=True)
                import tempfile
                temp_copy_path = tempfile.mkdtemp(prefix=f'{repo_name}_DEBUGEVAL_', dir=temp_dir)
                print('COPYING REPO to', temp_copy_path, '.......')
                shutil.copytree(src=repo_args['repo_path'], dst=temp_copy_path, dirs_exist_ok=True)
                if not temp_copy_path.endswith(os.sep):
                    temp_copy_path += os.sep
                tmp_running_path = temp_copy_path
                temp_running_path = os.path.join(tmp_running_path, repo_args["relative_running_path"])

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                # 修改Sourse_Copy中的文件
                testcases = {}
                if not os.path.exists(os.path.join(test_case_dir, 'testcases_info.jsonl')):
                    continue
                if os.path.getsize(os.path.join(test_case_dir, 'testcases_info.jsonl')) == 0:
                    continue
                with open(os.path.join(test_case_dir, 'testcases_info.jsonl'),'r') as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        if 'func' in data and 'id' in data and 'code' in data:
                            testcases[data['func']] = (data['id'], data['code'])

                if os.path.exists(os.path.join(test_case_dir, f'testcases_valid_{save_name}_info.jsonl')):
                    os.remove(os.path.join(test_case_dir, f'testcases_valid_{save_name}_info.jsonl'))


                with open(source_code_path, 'r') as f:
                    source_code = f.read().splitlines()
                base_passed = -1

                with open(os.path.join(test_case_dir, f'testcases_valid_{save_name}_info.jsonl'), 'w', encoding='utf-8') as file:
                    for func, item in testcases.items():
                        test_id, testcase = item
                        shutil.copy(source_code_path, source_code_path.replace(repo_path, tmp_running_path))
                        print('FUNC',func)
                        file_exists = os.path.exists(os.path.join(result_dir,f'''{func}_completed_{save_name}.py'''))
                        if not file_exists or True:
                            completed_key_block = "_ = 1"
                            
                            # 对齐缩进
                            prefix = source_code[:testcase['key_block_start_lineno'] -1]
                            suffix = source_code[testcase['key_block_end_lineno']:]
                            indented_completed_key_block = utils.align_indent(completed_key_block.splitlines(), source_code[testcase['key_block_start_lineno']-1:testcase['key_block_end_lineno']], prefix, suffix )
                            compeletd_code = prefix + indented_completed_key_block + suffix

                            with open(os.path.join(result_dir,f'''{func}_completed_{save_name}.py'''),'w') as f:
                                f.write('#test_copy\n' + '\n'.join(compeletd_code))
                            
                            shutil.copy(os.path.join(result_dir,f'''{func}_completed_{save_name}.py'''), source_code_path.replace(repo_path, tmp_running_path))
                    
                            os.chdir(tmp_running_path)
                            
                            BASH = f'''http_proxy=http://10.217.142.137:8080  https_proxy=http://10.217.142.137:8080 PYTHONPATH={temp_running_path} timeout 60 pytest {test_path} --tb=long > {result_dir}/{func}_{save_name}.log'''    
                            print(BASH)
                            status = os.system(BASH)
                            failed = 0
                            base_passed = 0
                            if status == 0:
                                base_passed, skipped, failed = read_log(os.path.join(result_dir, f'''{func}_retest.log'''))
                            print(f"passed: {passed}, base_passed: {base_passed}")

                if base_passed >= passed or base_passed == -1:
                    continue

                content = {
                    "id": id,
                    "project": repo_name,
                    "func": func_name,
                    "origin_file":origin_file.replace(copy_running_path, ""),
                    "test_list":[test_file.replace(copy_path, "")],
                    "prob_info": {
                        "func_start_lineno": func_lineno[0], 
                        "func_end_lineno": func_lineno[1], 
                        "key_block_start_lineno": start_line_no+code['func_start_lineno']-1, 
                        "key_block_end_lineno": end_line_no+code['func_start_lineno']-1, 
                        "new_func_code": new_code
                    },
                    "pytest_info": {
                        "total_num": passed, 
                        "base_passed_num": base_passed
                    }, 
                    "score": {"readability_score": "null", "accuracy_score": "null", "completeness_score": "null"},
                    "LLM_score": LLM_Score,
                    "type": "TDD", 
                    "language": "Python", 
                    "gen_model": "gpt4o", 
                    "is_difficult": ""

                    # "source_dir": sub_child.get("source_dir"),
                    # "call_position": sub_child.get("call_position"),
                    # 'class_start_lineno': class_lineno[0],
                    # 'class_end_lineno': class_lineno[1],
                    # 'func_start_lineno': func_lineno[0],
                    # 'func_end_lineno': func_lineno[1],
                    # 'key_block_start_lineno': start_line_no+code['func_start_lineno']-1,
                    # 'key_block_end_lineno': end_line_no+code['func_start_lineno']-1,
                    # 'new_func_code': new_code,
                    # "LLM_score": LLM_Score
                    # 'func_code': func_code,
                    # 'block_info': generate_code_blocks_dict(func, func_code),
                    # 'import_code': import_code
                }
                third_level_content.append(content)
                
                content_jsonl = json.dumps(content, ensure_ascii=False)
                if os.path.exists(TDD_results_path):
                    with open(TDD_results_path, 'r', encoding='utf-8') as file:
                        existing_ids = {json.loads(line)['id'] for line in file}
                    if content['id'] in existing_ids:
                        print(f"ID {content['id']} already exists, skipping.")
                    else:
                        with open(TDD_results_path, 'a', encoding='utf-8') as file:
                            file.write(content_jsonl + '\n')
                else:
                    with open(TDD_results_path, 'a', encoding='utf-8') as file:
                        file.write(content_jsonl + '\n')

            elif source_dir not in origin_files:
                print(f"not in source_dir: {source_dir}")
                    
    
    # 将结果写入到新的JSON文件
    with open(output_file_path, 'w') as output_file:
        json.dump(third_level_content, output_file, indent=4)


mapping_jsonl_path = repo_args["testcase_path"] + f'/{repo_name}/output_testcase_mapping_valid.jsonl'
repo_json_path = repo_args["root_path"] + f'func_testcases/{repo_name}/'
TDD_results_path = repo_args["root_path"] + f'TDD_results/{repo_name}/{repo_name}.jsonl'
os.makedirs(os.path.dirname(TDD_results_path), exist_ok=True)

def execute_function_on_matching_folders(repo_json_path, mapping_jsonl_path, TDD_results_path):
    print("Running execute_function_on_matching_folders......")
    for root, dirs, files in os.walk(repo_json_path):
        if 'funcCallTree_new.json' in files:
            extract_third_level_content(root, mapping_jsonl_path, TDD_results_path)

execute_function_on_matching_folders(repo_json_path, mapping_jsonl_path, TDD_results_path)