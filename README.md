
# VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models

<a href='https://arxiv.org/abs/2505.08455'><img src='https://img.shields.io/badge/arXiv-paper-red'></a>
<a href='https://pritamqu.github.io/VCRBench/'><img src='https://img.shields.io/badge/project-VCRBench-blue'></a> 
<a href='https://huggingface.co/datasets/pritamqu/VCRBench'><img src='https://img.shields.io/badge/huggingface-datasets-green'></a> 
<a href='https://github.com/pritamqu/VCRBench'><img src='https://img.shields.io/badge/github-repository-purple'></a> 
<a href='https://pritamqu.github.io/VCRBench/#leaderboard'><img src='https://img.shields.io/badge/vcrbench-leaderboard-yellow'></a> 

Authors: [Pritam Sarkar](https://pritamsarkar.com) and [Ali Etemad](https://www.aiimlab.com/ali-etemad)

This repository provides the official implementation of **[VCRBench](https://arxiv.org/abs/2505.08455)**.

## Installation

Clone the repository and navigate to the VCRBench directory:

```
git clone https://github.com/pritamqu/VCRBench
cd VCRBench
```

This repository supports several LVLMs for direct evaluation on VCRBench.


## Download VCRBench

Our data can be accessed via this link: <a href='https://huggingface.co/datasets/pritamqu/VCRBench'>[VCRBench]</a>

Please download the videos and questions from the link and save them in your local directory.

```
mkdir HF_DATA # create a dir where you want to download the data
cd HF_DATA # go to that dir
git lfs install
git clone https://huggingface.co/datasets/pritamqu/VCRBench
```

Please make sure to update the `video-folder` and `question-file` in inference scripts as per your path.

## Leaderboard

See our leaderborad [here](https://pritamqu.github.io/VCRBench/#leaderboard).
If you want to add your model to our leaderboard, please send model responses to `pritam.sarkar@queensu.ca`, in the same the format as provided in [sample response](./output/random/response_1.json).

## Download model weights

You can download the open-source weights using:

```
git lfs install
git clone git@hf.co:Qwen/Qwen2.5-VL-72B-Instruct
```

OR you can also evaluate models using API as done for [Gemini](./models/gemini.py) and [GPT4o](./models/gpt.py).

### Setting up the environment

```
conda create -n vcr python=3.10 -y
conda activate vcr
pip install -r requirements.txt
```

## Evaluating on VCRBench

We provide scripts to directly evaluate several open-source (e.g., Qwen2.5-VL-Instruct, InternVL2_5, VideoLLaMA3, VideoLLaVA) and closed-source (e.g., Gemini, GPT-4o) models on VCRBench.  
Evaluation scripts are located [here](./scripts/). For example, to evaluate Qwen2.5-VL-72B-Instruct:

```
bash scripts/qwen_2_5_vl/inference72.sh
```

You can use the given evaluation scripts as a reference to evaluate on other models. 

## Evaluating on VCRBench equipped with RRD

We also provide scripts to test open-source models equipped with RRD.  
For example, to evaluate Qwen2.5-VL-72B-Instruct with RRD:

```
bash scripts/qwen_2_5_vl/rrd72.sh
```

## Citation

If you find this work useful, please consider citing our paper:

```
@misc{sarkar2025vcrbench,
      title={VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models}, 
      author={Pritam Sarkar and Ali Etemad},
      year={2025},
      eprint={2505.08455},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## Usage and License Notices

This project incorporates datasets and model checkpoints that are subject to their respective original licenses.  
Users must adhere to the terms and conditions specified by these licenses.

Assets used in this work include, but are not limited to, [CrossTask](https://github.com/DmZhukov/CrossTask).  
This project does not impose any additional constraints beyond those stipulated in the original licenses. Users must ensure their usage complies with all applicable laws and regulations.

This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---
For any issues or questions, please open an issue or contact **Pritam Sarkar** at pritam.sarkar@queensu.ca!
