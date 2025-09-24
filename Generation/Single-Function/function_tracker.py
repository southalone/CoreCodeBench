import os
import shutil
import subprocess
import json
import unittest
import pycallgraph
import argparse
import utils

default_repo_name = 'transformers'
default_file_path = 'src/transformers'
default_file_name = "image_transforms"
repo_args = utils.get_repo_args(default_repo_name)
root_path = repo_args["root_path"]
default_test_path = root_path + "Source_Copy/transformers/tests/test_image_transforms.py"
default_output_dir = root_path

parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', default=default_repo_name, help='Repository name')
# parser.add_argument('--file_path', default=default_file_path, help='File path')
# parser.add_argument('--file_name', default=default_file_name, help='File name')
parser.add_argument('--test_path', default=default_test_path, help='Test path')
parser.add_argument('--output_dir', default=default_output_dir, help='Output directory')
args = parser.parse_args()

def copy_and_modify_test_file(original_file, copy_file):
    # pass
    # 复制文件到新的路径
    shutil.copyfile(original_file, copy_file)
    with open(copy_file, "r") as cfile:
        code = cfile.read()
    unittest_code = """if __name__ == "__main__":
    unittest.main()"""
    unittest_code_2 = """if __name__ == '__main__':
    unittest.main()"""
    # print(code)
    if unittest_code in code:
        print(f"!!!!!!!!exists unitest main!!!!!!!!")
        new_code = code.replace(unittest_code, "\n")
        with open(copy_file, "w") as cfile:
            cfile.write(new_code)
    if unittest_code_2 in code:
        print(f"!!!!!!!!exists unitest main!!!!!!!!")
        new_code = code.replace(unittest_code_2, "\n")
        with open(copy_file, "w") as cfile:
            cfile.write(new_code)


    import_code = ""
    run_code = ""
    if 'import unittest' in code:
        import_code = "import unittest"
        run_code = "unittest.main(exit=False)"
    else:
        import_code = "import pytest"
        run_code = "pytest.main([__file__, '-s'])"
    if args.import_name == "":
        import_line = ""
        project_root = f"\"{args.copy_running_path}\""
    else:
        import_line = f"import {args.import_name}"
        project_root = f"os.path.dirname(os.path.dirname({args.import_name}.__file__))"
    ban_code = ""
    if args.repo_name == "langchain_core":
        ban_code = f"""from blockbuster import BlockBuster
    BlockBuster.deactivate
"""
    
    # 在复制的文件中插入代码
    with open(copy_file, 'a') as f:
        f.write(f'''\n
if __name__ == "__main__":
    # 事先要设置PYTHONPATH = transformers src 的路径
    from pycallgraph import PyCallGraph, Config
    from pycallgraph import GlobbingFilter
    from pycallgraph.output import GraphvizOutput

    from pycallgraph.util import CallNode, CallNodeEncoder
    import os
    import sys
    print(sys.path)
    import json
    {import_line}
    {import_code}
    {ban_code}

    # 定义要测试的文件路径
    project_root = {project_root}
    print(project_root)
    print(sys.path)
    sys.path.append(project_root)

    config = Config(project_root=project_root)
    config.trace_filter = GlobbingFilter(
        include=[
            '{args.repo_name}.*',   # 包括项目中的所有模块
            '{args.import_name}.*',
        ],
        exclude=[
            'pycallgraph.*', # 排除 pycallgraph 自身
            'os.*',          # 排除 os 模块
            'sys.*',         # 排除 sys 模块
            '*.<listcomp>*','*.<dictcomp>*','*.<setcomp>*','*.<genexpr>*','*.<module>*','*.<locals>*','*.<lambda>*'
        ]
    )

    # 使用 PyCallGraph 进行调用跟踪
    with PyCallGraph(output=GraphvizOutput(),config=config) as pycg:
        {run_code}
    
    # 获取调用树
    call_tree = pycg.get_call_tree()
    
    def serialize_tree(node, depth):
        if depth == 6:
            return None
        return {{
            'name': node.name,
            'source_dir': node.source_dir,
            'call_position': node.call_position,
            'children': [serialize_tree(child, depth+1) for child in node.children],
        }}

    def merge_trees(root):
        # 使用字典存储合并后的节点
        node_dict = {{}}
        def merge_node(node):
            if node.name not in node_dict:
                # 如果节点名称不存在，则创建一个新的节点并添加到字典中
                new_node = CallNode(node.name, node.source_dir, node.call_position)
                node_dict[node.name] = new_node
            else:
                # 如果节点名称已经存在，则获取现有节点
                new_node = node_dict[node.name]
            
            for child in node.children:
                merged_child = merge_node(child)
                # 检查子节点是否已经存在于当前节点的子节点列表中
                if merged_child not in new_node.children and merged_child.name != new_node.name:
                    new_node.add_child(merged_child)

            return new_node
        # 从根节点开始合并
        new_root = merge_node(root)
        return new_root
    
    def get_2level_tree(root, level):
        # 把原来的树剪枝，只保留2层
        if level == 2:
            root.children = []
            return
        else:
            # level = 0/1
            for child in root.children:
                get_2level_tree(child, level+1)


    merged_tree = merge_trees(call_tree)
    get_2level_tree(call_tree, 0)
    
    # 保存调用树到JSON文件
    with open('{os.path.join(args.test_case_dir, 'funcCallTree_new.json')}', 'w') as output_file:
        json.dump(serialize_tree(merged_tree, 0), output_file, indent=4)
    # 保存node_dict到JSON文件
    with open('{os.path.join(args.test_case_dir, 'funcCallTree2level.json')}', 'w') as output_file:
        json.dump(call_tree, output_file, cls=CallNodeEncoder, indent=4)
''')
# import pytest
# pytest.main([__file__])

def track_function(repo_name, test_path, output_dir):
    args.repo_name = repo_name
    args.test_path = test_path
    args.output_dir = output_dir
    
    repo_args = utils.get_repo_args(repo_name)
    repo_path = repo_args['repo_path']
    repo_running_path = repo_args['repo_running_path']
    copy_path = repo_args["copy_path"]
    import_name = repo_args["import_name"]
    args.import_name = import_name


    copy_running_path = repo_running_path.replace(repo_path, copy_path)
    args.copy_running_path = copy_running_path
    test_case_dir = os.path.join(output_dir, 'func_testcases', repo_name, test_path.replace(copy_path, "").replace(".py", ""))
    if not os.path.exists(test_case_dir):
        os.makedirs(test_case_dir)
    print(test_case_dir)
    args.test_case_dir = test_case_dir
    copy_test_path = test_path.replace(repo_path, copy_path)
    # original_file_copy.py
    copy_file = copy_test_path.replace('.py', '_copy.py')
    repo_test_path = repo_args["test_path"]
    init_path = os.path.join(repo_test_path, "__init__.py")
        
    # 复制并修改文件
    copy_and_modify_test_file(copy_test_path, copy_file)
    env = os.environ.copy()
    env["PYTHONPATH"] = copy_running_path
    env["http_proxy"] = "http://10.217.142.137:8080"
    env["https_proxy"] = "http://10.217.142.137:8080"
    module_name = copy_file.replace(copy_path,'').replace('.py','').replace('/','.')
    if not os.path.exists(test_case_dir.replace("func_testcases", "func_testcases/log")):
        os.makedirs(test_case_dir.replace("func_testcases", "func_testcases/log"))
    with open(os.path.join(test_case_dir.replace("func_testcases", "func_testcases/log"), "log_test.txt"), "w") as output_file:
        
        try:
            if os.path.exists(init_path):
                command = [
                    f'PYTHONPATH=\"{copy_running_path}\"', "http_proxy=\"http://10.217.142.137:8080\"", "https_proxy=\"http://10.217.142.137:8080\"", 'python', '-m', f'{module_name}'
                ]
                print(' '.join(command))
                subprocess.run(['python', '-m', module_name], cwd=copy_path, env=env, stdout=output_file, stderr = output_file)
            else:
                command = [
                    f'PYTHONPATH=\"{copy_running_path}\"', "http_proxy=\"http://10.217.142.137:8080\"", "https_proxy=\"http://10.217.142.137:8080\"", 'python', f'{copy_file}'
                ]
                print(' '.join(command))
                subprocess.run(['python', copy_file], cwd=copy_path, env=env, stdout=output_file, stderr = output_file)
        except Exception as e:
            import traceback
            log_dir = test_case_dir.replace("func_testcases", "func_testcases/log")
            print(log_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            
            command = [
                f'PYTHONPATH=\"{copy_running_path}\"', "http_proxy=\"http://10.217.142.137:8080\"", "https_proxy=\"http://10.217.142.137:8080\"", 'python', '-m', f'{module_name}'
            ]
            with open (os.path.join(log_dir, "function_tracker_subprocess.txt"), "w") as output_file:
                traceback.print_exc(file=output_file)
                traceback.print_exc()
                print("something went wrong")
                output_file.write(f"An exception occurred: {e.__class__.__name__}: {e}\n")
                output_file.write("command: \n" + ' '.join(command))
            traceback.print_exc()     
            return False
        # print(command)
    return True     

if __name__ == "__main__":
    track_function(args.repo_name, args.test_path, args.output_dir)
