# LMR-BENCH: Evaluating LLM Agent’s Ability on Reproducing Language Modeling Research


## Datasets & Scripts.
Repo to be revised. Coming soon.

### Abstract: 
 Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present \ours, a comprehensive benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from NLP research papers. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked methods, and instructions for implementing these methods.
We conduct extensive experiments in standalone and agent-based settings on state-of-the-art LLMs, evaluating the accuracy of unit tests and performing both LLM and human evaluation of code correctness.
Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLMs’ ability to autonomously reproduce scientific research. We will release our benchmark and code after publication.


## Environment
python $\geq$ 3.12
```
pip install -r requirements.txt
```

## Benchmark access
<!-- The benchmark used in our paper can be downloaded in https://drive.google.com/drive/folders/1bkSx0ml4VobEV2bDfcrFdvi51yC5vSfu?usp=drive_link.

The full benchmark will be updated in https://drive.google.com/drive/folders/1bkSx0ml4VobEV2bDfcrFdvi51yC5vSfu?usp=drive_link. -->







## Generation
### OpenHands
#### Setup Environment and LLM Configuration
Please follow instruction [here](https://github.com/All-Hands-AI/OpenHands/blob/main/evaluation/README.md#setup) to setup your local development environment and LLM. 

We have not integrated our benchmark into OpenHands. So after finishing set up the develop environment, copy the downloaded benchmark before into the folder lmrbench/benchmark and then copy the folder lmrbench under OpenHands/evaluation/benchmark/datasets.

The structure of the directory of lmrbench should be like the following:
```text
lmrbench/
├── benchmark/
│   └── project_folder_1/
│       ├── repository_folder_1/
│       │   ├── ...
│       │   └── unit_test/
│       │       └── unit_test_1.py
│       ├── info.json
│       ├── Dockerfile
│       └── golden_files/
│           └── ...
└── scripts/
    └── ...
```

#### Run Inference on LMR-Bench
./evaluation/benchmarks/lmrbench/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [num_workers] [eval_output_dir] [cache_path] [dest_path]

eval_output_dir: Path to store OpenHands Agent's generation logs
cache_path: Path to store OpenHands Agent's detailed events and other cache. It can be same as eval_output_dir
dest_path: Path to store the repositories after OpenHands Agent's revision.

##### Example
./evaluation/benchmarks/lmrbench/scripts/run_infer.sh llm.eval_gpt4o "" "" "" "" [$LOG_DIR] [$LOG_DIR] [$DEST_PATH]


<!-- The output_path above only saves logs of the agent. To save repositories revised by OpenHands, we need to revise line 68 and 162 in run_infer.py.
The revised repositories will be saved in the folder written in line 162. -->


### No Agent
sh scripts/no_agent_generation.sh [$DATA_FOLDER] [$DEST_PATH]

<!-- example:
```
sh scripts/no_agent_generation.sh [dataset_path] [output_repository_path]
``` -->

## Evaluation
### Unit test evaluation
sh scripts/unit_test_evaluation.sh [$DEST_PATH] [$EVAL_RESULT_PATH]

<!-- example:
```
sh scripts/base_agent_generation.sh /home/sxy240002/research_agent/NLPBench/benchmark/datasets_final /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o
``` -->

### LLM-as-a-judge evaluation
sh scripts/llm_as_a_judge_evaluation.sh [$DEST_PATH] [$EVAL_RESULT_PATH]

<!-- example:
```
sh scripts/llm_as_a_judge_evaluation.sh /home/sxy240002/research_agent/NLPBench/outputs/BaseAgent/gpt4o /home/sxy240002/research_agent/NLPAgentBench/llm_as_a_judge_evaluation_results/BaseAgent/gpt4o
``` -->


## Analysis

### Data contamination
For performance of each sample, it is saved in the unit_test_evaluation and llm_as_a_judge folder.
To calculate the similarity score, we compute a model’s familiarity with a given document as the mean probability a model assigns to each token in that document, conditional on all preceding tokens(MLE-Bench). We calculate the familarity with the goal file(file path is info.json for each paper).