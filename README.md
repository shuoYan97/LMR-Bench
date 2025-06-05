# LMR-BENCH: Evaluating LLM Agent’s Ability on Reproducing Language Modeling Research


## Overview
*LMR-Bench* is a benchmark for evaluating large language model (LLM) agents on their ability to reproduce code from *NLP research papers*.
Given a *research paper*, a *code repo with masked methods*, and a *implementation instruction*, an LLM agent is tasked with generating *patch* code to correctly fill in the missing methods. The benchmark covers 28 tasks from 23 top-tier NLP papers across nine research categories, providing a systematic way to assess the scientific reasoning and code synthesis abilities of LLMs.


### Abstract: 
 Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present \ours, a comprehensive benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from NLP research papers. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked methods, and instructions for implementing these methods.
We conduct extensive experiments in standalone and agent-based settings on state-of-the-art LLMs, evaluating the accuracy of unit tests and performing both LLM and human evaluation of code correctness.
Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLMs’ ability to autonomously reproduce scientific research. We will release our benchmark and code after publication.


## Environment Setup
LMR-Bench requires Python ≥ 3.12.

We recommend using a virtual environment to avoid dependency conflicts.

**1. Clone the repository**:
```
git clone git@github.com:du-nlp-lab/LMR-Bench.git
cd LMR-Bench
```

The structure of the directory of LMR-Bench should be like the following:
```text
LMR-Bench/
├── benchmark/                        # Contains all datasets. Each project has a subfolder.
│   └── project_folder_1/
│       ├── repository_folder_1/
│       │   ├── ...                   # Masked code repo (Rewrite & Mask)
│       │   └── unit_test/
│       │       └── unit_test_1.py    # Unit test file (Annotated)
│       ├── info.json                 # Metadata for this benchmark task. (Annotated)
│       ├── Dockerfile                # Docker ENV for reproducibility. We've built and pushed Docker Images for each project to DOCKER HUB for faster evaluation.
│       └── golden_files/
│           └── ...                   # Reference solutions or golden outputs
│
├── evaluation/                       # Evaluation
│   ├── ...                 
├── generation/                       # Generation/inference (noagent & necessary scripts for OpenHands)
│   ├── NoAgent/
│   └── OpenHands/
├── results/                      
│   ├── human_evaluation/
│   ├── llm_as_a_judge_evaluation/
│   ├── logs/
│   └── unit_test_evaluation/
├── scripts/                          # Shell scripts
│   ├── no_agent_generation.sh
│   ├── no_agent_generation_claude.sh
│   ├── llm_as_a_judge_evaluation.sh
│   ├── unit_test_evaluation.sh
│   └── unit_test_evaluation_golden.sh
├── utils/                            # Utility modules
│   └── ...
└── requirements.txt

```

- `benchmark/`: Contains all benchmark tasks. Each project is a subdirectory with its repository, metadata, unit tests, and golden/reference files.
- `evaluation/`: Main Python evaluation logic.
- `generation/`: Code for generation/inference (with or without agents), organized by method category. Add your agent under this folder if you want to test your performance. 
- `results/`: Output/results directories.
- `scripts/`: Shell scripts for automated/batch execution for running generation and evaluation.
- `utils/`: Utility modules and functions.

**2. Install dependencies**:
```
pip install -r requirements.txt
```

<!-- 3. **Download the benchmark:** The benchmark data used in our paper can be downloaded from [this link](https://drive.google.com/drive/folders/1bkSx0ml4VobEV2bDfcrFdvi51yC5vSfu?usp=drive_link). -->

**Test your installation**:
Run a sample evaluation script to ensure everything works:

```
python evaluation/unit_test_evaluation.py \
    --output_repository_path generation/noagent/sample_repo \
    --unit_test_evaluation_path results/unit_test_evaluation/sample_result
```



## ⚡ Generation

### 🔧 OpenHands

#### Environment and LLM Setup

Please follow the official setup instructions in the [OpenHands repository](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/README.md#setup) to configure your local development environment and LLM.

#### Preparing LMR-Bench for OpenHands

> **Integration steps:**  
> 1. **Copy code:**  
>    - Copy the folder `LMR-Bench/generation/OpenHands/evaluation/benchmarks/lmr_bench` into the corresponding path inside your OpenHands repository.
> 2. **Copy benchmark data:**  
>    - Copy `LMR-Bench/benchmark` under `OpenHands/evaluation/benchmarks/lmr_bench/` for evaluation.
>
> Your final structure in OpenHands should look like:
> ```
> OpenHands/
> └── evaluation/
>     └── benchmarks/
>         └── lmr_bench/
>             ├── <copied benchmark data>
>             └── <code and scripts>
> ```

#### Run Inference on LMR-Bench

```bash
./evaluation/benchmarks/lmrbench/scripts/run_infer.sh \
    [MODEL_CONFIG] [GIT_VERSION] [AGENT] [EVAL_LIMIT] [NUM_WORKERS] \
    [EVAL_OUTPUT_DIR] [CACHE_PATH] [DEST_PATH]
```

EVAL_OUTPUT_DIR: Path to store OpenHands Agent's generation logs  
CACHE_PATH: Path to store OpenHands Agent's detailed events and other cache. It can be same as EVAL_OUTPUT_DIR  
DEST_PATH: Path to store the repositories after OpenHands Agent's revision.  

##### Example
```
./evaluation/benchmarks/lmrbench/scripts/run_infer.sh llm.eval_gpt4o "" "" "" "" [LOG_DIR] [LOG_DIR] [DEST_PATH]
```


<!-- The output_path above only saves logs of the agent. To save repositories revised by OpenHands, we need to revise line 68 and 162 in run_infer.py.
The revised repositories will be saved in the folder written in line 162. -->


### No Agent
```
sh scripts/no_agent_generation.sh [DATA_FOLDER] [DEST_PATH]
```


## Evaluation
### Unit test evaluation
```
sh scripts/unit_test_evaluation.sh [DEST_PATH] [EVAL_RESULT_PATH]
```

<!-- example:
```
sh scripts/base_agent_generation.sh /home/sxy240002/research_agent/NLPBench/benchmark/datasets_final /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o
``` -->

### LLM-as-a-judge evaluation
```
sh scripts/llm_as_a_judge_evaluation.sh [DEST_PATH] [EVAL_RESULT_PATH]
```

<!-- example:
```
sh scripts/llm_as_a_judge_evaluation.sh /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o /home/sxy240002/research_agent/NLPAgentBench/llm_as_a_judge_evaluation_results/BaseAgent/gpt4o
``` -->


## Analysis

### Data contamination
For performance of each sample, it is saved in the unit_test_evaluation and llm_as_a_judge folder.
To calculate the similarity score, we compute a model’s familiarity with a given document as the mean probability a model assigns to each token in that document, conditional on all preceding tokens(MLE-Bench). We calculate the familarity with the goal file(file path is info.json for each paper).
