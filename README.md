# CoreCodeBench
[CoreCodeBench](https://huggingface.co/datasets/tubehhh/CoreCodeBench-Single) is a configurable, multi-scenario repository-level benchmark designed to comprehensively evaluate the engineering capabilities of Large Language Models (LLMs) in real-world software development tasks. Built on the automated pipeline CorePipe, CoreCodeBench transforms open-source repositories into diverse, controllable, and reliable benchmark cases, covering development, bug fixing, and test-driven development (TDD) scenarios.

## Quick Start

### Generation
#### Preprocess
```
conda activate {repo_name_env}
./Generation/Single-Function/Preprocess.sh {repo_name}
```
#### Single Function Problem Generation
1. Development
    ```
    conda activate {repo_name_env}
    ./Generation/Single-Function/Development_generate.sh {repo_name}
    ./Generation/Single-Function/Filter.sh {repo_name} {model_name}
    ```
2. TDD
    ```
    conda activate {repo_name_env}
    ./Generation/Single-Function/TDD_generate.sh {repo_name}

    ```
3. Debug
    ```
    conda activate {repo_name_env}
    ./Generation/Single-Function/BugFix_generate.sh {repo_name} {gen_model} {rewrite_model}
    ```
#### Multi-Function Problem Generation
1. Development
    ```
    conda activate {repo_name_env}
    ./Generation/Multi-Function/function_generate.sh {repo_name}
    ```
2. TDD
    ```
    conda activate {repo_name_env}
    ./Generation/Multi-Function/function_generate_tdd.sh {repo_name}
    ```
3. BugFix
    ```
    conda activate {repo_name_env}
    ./Generation/Multi-Function/function_generate_debug.sh {repo_name}
    ```
4. Difficult
    ```
    conda activate {repo_name_env}
    ./Generation/Multi-Function/function_generate_difficult.sh {repo_name}
    ```

### Evaluation
#### Single Function Problem Evaluation
1. Development
    ```
    conda activate {repo_name_env}
    ./Evaluation/Single-Function/Development_evaluate.sh
    ```
2. BugFix
    ```
    conda activate {repo_name_env}
    ./Evaluation/Single-Function/Debug_evaluate.sh
    ```
3. TDD
    ```
    conda activate {repo_name_env}
    ./Evaluate/Single-Function/TDD_evaluate.sh
    ```
#### Multi Function Problem Evaluation
1. Development
    ```
    conda activate {repo_name_env}
    ./Evaluation/Multi-Function/function_test_run.sh {repo_name} {model_name}
    ```
2. TDD
    ```
    conda activate {repo_name_env}
    ./Evaluation/Multi-Function/function_test_tdd_run.sh {repo_name} {model_name}
    ```
3. BugFix
    ```
    conda activate {repo_name_env}
    ./Evaluation/Multi-Function/function_test_debug_run.sh {repo_name} {model_name}
    ```
4. Difficult
    ```
    conda activate {repo_name_env}
    ./Evaluation/Multi-Function/function_test_difficult_run.sh {repo_name} {model_name}
    ```

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please open an issue or contact fulingyue@sjtu.edu.cn.