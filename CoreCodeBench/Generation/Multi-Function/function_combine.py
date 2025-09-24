import json
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--repo_name', type=str, default='transformers', help='Repository name')
parser.add_argument('--if_comments', type=str, default='full', help='Repository name')

args = parser.parse_args()

def merge_jsonl_data(file_path):
    merged_data = {}

    # 读取jsonl文件
    sum = 0
    with open(file_path, 'r') as infile:
        for line in infile:
            sum += 1
            entry = json.loads(line)
            ids_tuple = tuple(entry["id"])

            if ids_tuple not in merged_data:
                merged_data[ids_tuple] = {
                    "project": entry["project"],
                    "origin_file": [],
                    "test_list": [],
                    "prob_info": [],
                    "type": [],
                    "node": [],
                    "test": [],
                    "language": entry["language"],
                    "toolfunc_count": 0,
                    "func_count": 0,
                    "pytest_info": {"total_num": 0, "base_passed_num": 0}
                }
            else:
                print(entry["test_list"][0])

            # 遍历当前 entry 的数据并合并到结果中
            for i, node in enumerate(entry["node"]):
                if node not in merged_data[ids_tuple]["node"]:
                    merged_data[ids_tuple]["node"].append(node)
                    merged_data[ids_tuple]["origin_file"].append(entry["origin_file"][i])
                    merged_data[ids_tuple]["prob_info"].append(entry["prob_info"][i])
            
            merged_data[ids_tuple]["test_list"].append(entry["test_list"][0])
            
            merged_data[ids_tuple]["type"] = entry["type"]
            merged_data[ids_tuple]["test"].extend(entry["test"])
            merged_data[ids_tuple]["toolfunc_count"] = entry["toolfunc_count"]
            merged_data[ids_tuple]["func_count"] = entry["func_count"]
            merged_data[ids_tuple]["pytest_info"]["total_num"] += entry["pytest_info"]["total_num"]
            merged_data[ids_tuple]["pytest_info"]["base_passed_num"] += entry["pytest_info"]["base_passed_num"]

    # 去除 type 和 test 的重复项，保持原有顺序
    print("!!!Running combine\n\n")
    print(sum)
    print(len(merged_data))
    for ids, info in merged_data.items():
        info["test"] = list(dict.fromkeys(info["test"]))

    # 将结果写回jsonl文件
    if args.if_comments == "full":
        combine_path = f'./func_testcases/{args.repo_name}/func_testcases_combine_info.jsonl'
    else:
        combine_path = f'./func_testcases/{args.repo_name}/func_{args.if_comments}_testcases_combine_info.jsonl'

    with open(combine_path, 'w') as outfile:
        for ids, info in merged_data.items():
            result_entry = {
                "id": list(ids),
                "project": info["project"],
                "origin_file": info["origin_file"],
                "test_list": info["test_list"],
                "prob_info": info["prob_info"],
                "type": info["type"],
                "node": info["node"],
                "test": info["test"],
                "language": info["language"],
                "toolfunc_count": info["toolfunc_count"],
                "func_count": info["func_count"],
                "pytest_info": {"total_num": info["pytest_info"]["total_num"], "base_passed_num": info["pytest_info"]["base_passed_num"]}
            }
            outfile.write(json.dumps(result_entry) + '\n')

if args.if_comments == "full":
    valid_path = f'./func_testcases/{args.repo_name}/func_testcases_valid_info.jsonl'
else:
    valid_path = f'./func_testcases/{args.repo_name}/func_{args.if_comments}_testcases_valid_info.jsonl'
merge_jsonl_data(valid_path)
