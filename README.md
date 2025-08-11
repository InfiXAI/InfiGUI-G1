<h1 align="center">
<img src="assets/infigui-g1-logo.png" width="100" alt="InfiGUI-G1" />
<br>
InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2508.05731"><img src="https://img.shields.io/badge/arXiv-Preprint-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
  <a href="https://huggingface.co/papers/2508.05731"><img src="https://img.shields.io/badge/HuggingFace-Daily%20Papers-ff9800?style=flat&logo=huggingface" alt="Hugging Face Paper"></a>
  <a href="https://huggingface.co/InfiX-ai/InfiGUI-G1-3B"><img src="https://img.shields.io/badge/Model-InfiGUI--G1--3B-007ec6?style=flat&logo=huggingface" alt="InfiGUI-G1 3B Model"></a>
  <a href="https://huggingface.co/InfiX-ai/InfiGUI-G1-7B"><img src="https://img.shields.io/badge/Model-InfiGUI--G1--7B-007ec6?style=flat&logo=huggingface" alt="InfiGUI-G1 7B Model"></a>
</p>

<p align="center">
  This is the official repository for the paper <a href="https://arxiv.org/abs/2508.05731">InfiGUI-G1</a>.
  <br>
  <strong>InfiGUI-G1 enhances GUI grounding with Adaptive Exploration Policy Optimization (AEPO) to overcome exploration bottlenecks.</strong>
</p>

## 🌟 Overview

A fundamental challenge for GUI agents is robustly grounding natural language instructions, which requires not only precise **spatial alignment** (locating elements accurately) but also correct **semantic alignment** (identifying the functionally appropriate element). While existing Reinforcement Learning with Verifiable Rewards (RLVR) methods have enhanced spatial precision, they often suffer from inefficient exploration. This "confidence trap" bottlenecks semantic alignment, preventing models from discovering correct actions for difficult semantic associations.

To address this critical exploration problem, we introduce **InfiGUI-G1**, a series of models trained with **Adaptive Exploration Policy Optimization (AEPO)**. AEPO overcomes the exploration bottleneck by integrating a **multi-answer generation** strategy to explore a diverse set of candidate actions in a single forward pass. This exploration is guided by a theoretically-grounded **Adaptive Exploration Reward (AER)** function, derived from first principles of efficiency ($\eta=U/C$), which provides rich, informative learning signals to dynamically balance exploration and exploitation.

<div align="center">
  <img src="assets/methodology.png" width="95%" alt="AEPO Framework">
  <p><i>Comparison between a naive RL baseline and our AEPO framework. AEPO's multi-answer generation and adaptive reward mechanism break the exploration bottleneck, enabling robust semantic alignment by deriving an informative learning signal.</i></p>
</div>

## 🔥 News
- 🔥 ***`2025/08/11`*** Our paper "[InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](https://arxiv.org/abs/2508.05731)" released.
- 🔥 ***`2025/05/15`*** Our paper "[OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use](https://os-agent-survey.github.io/)" is accepted by *ACL 2025*.
- 🔥 ***`2025/4/19`*** Our paper "[InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners](https://arxiv.org/abs/2504.14239)" released.
- 🔥 ***`2025/1/9`*** Our paper "[InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection](https://arxiv.org/abs/2501.04575)" released.
- 🔥 ***`2024/12/12`*** Our paper "[OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use](https://os-agent-survey.github.io/)" released.
- 🔥 ***`2024/4/2`*** Our paper "[InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks](https://infiagent.github.io/)" is accepted by *ICML 2024*.

## 🚀 Updates

- 🚀 ***`2025/08/11`*** The evaluation script is now available. See the [Evaluation](#️-evaluation) section for details.
- 🚀 ***`2025/08/11`*** The models [InfiGUI-G1-3B](https://huggingface.co/InfiX-ai/InfiGUI-G1-3B) and [InfiGUI-G1-7B](https://huggingface.co/InfiX-ai/InfiGUI-G1-7B) are now publicly available on Hugging Face.
- 🚀 ***`2025/08/08`*** The official repository for InfiGUI-G1 is now public.

## 🗺️ Roadmap

- ✅ InfiGUI-G1-3B Model Release
- ✅ InfiGUI-G1-7B Model Release
- ✅ Evaluation Code and Instructions
- ⏳ Training Code and Scripts

## 📊 Results

Our InfiGUI-G1 models, trained with the AEPO framework, establish new state-of-the-art results among open-source models across a diverse and challenging set of GUI grounding benchmarks.

### MMBench-GUI (L2) Results

On the comprehensive MMBench-GUI benchmark, which evaluates performance across various platforms and instruction complexities, our InfiGUI-G1 models establish new state-of-the-art results for open-source models in their respective size categories.

<div align="center">
  <img src="assets/results_mmbench-gui.png" width="90%" alt="MMBench-GUI Results">
</div>

### ScreenSpot-Pro Results

On the challenging ScreenSpot-Pro benchmark, designed to test semantic understanding on high-resolution professional software, InfiGUI-G1 demonstrates significant improvements, particularly on icon-based grounding tasks. This highlights AEPO's effectiveness in enhancing semantic alignment by associating abstract visual symbols with their functions.

<div align="center">
  <img src="assets/results_screenspot-pro.png" width="90%" alt="ScreenSpot-Pro Results">
</div>

### UI-Vision (Element Grounding) Results

InfiGUI-G1 shows strong generalization capabilities on the UI-Vision benchmark, which is designed to test robustness across a wide variety of unseen desktop applications. Achieving high performance confirms that our AEPO framework fosters a robust understanding rather than overfitting to the training data.

<div align="center">
  <img src="assets/results_ui-vision.png" width="90%" alt="UI-Vision Results">
</div>

### UI-I2E-Bench Results

To further probe semantic reasoning, we evaluated on UI-I2E-Bench, a benchmark featuring a high proportion of implicit instructions that require reasoning beyond direct text matching. Our model's strong performance underscores AEPO's ability to handle complex, indirect commands.

<div align="center">
  <img src="assets/results_i2e-bench.png" width="90%" alt="UI-I2E-Bench Results">
</div>

### ScreenSpot-V2 Results

On the widely-used ScreenSpot-V2 benchmark, which provides comprehensive coverage across mobile, desktop, and web platforms, InfiGUI-G1 consistently outperforms strong baselines, demonstrating the broad applicability and data efficiency of our approach.

<div align="center">
  <img src="assets/results_screenspot-v2.png" width="90%" alt="ScreenSpot-V2 Results">
</div>

## ⚙️ Evaluation

This section provides instructions for reproducing the evaluation results reported in our paper.

### 1. Getting Started

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/InfiXAI/InfiGUI-G1.git
cd InfiGUI-G1
```

### 2. Environment Setup

The evaluation pipeline is built upon the [vLLM](https://github.com/vllm-project/vllm) library for efficient inference. For detailed installation guidance, please refer to the official vLLM repository. The specific versions used to obtain the results reported in our paper are as follows:

- **Python**: `3.10.12`
- **PyTorch**: `2.6.0`
- **Transformers**: `4.50.1`
- **vLLM**: `0.8.2`
- **CUDA**: `12.6`

The reported results were obtained on a server equipped with 4 x NVIDIA H800 GPUs.

### 3. Model Download

Download the InfiGUI-G1 models from the Hugging Face Hub into the `./models` directory.

```bash
# Create a directory for models
mkdir -p ./models

# Download InfiGUI-G1-3B
huggingface-cli download --resume-download InfiX-ai/InfiGUI-G1-3B --local-dir ./models/InfiGUI-G1-3B

# Download InfiGUI-G1-7B
huggingface-cli download --resume-download InfiX-ai/InfiGUI-G1-7B --local-dir ./models/InfiGUI-G1-7B
```

### 4. Dataset Download and Preparation

Download the required evaluation benchmarks into the `./data` directory.

```bash
# Create a directory for datasets
mkdir -p ./data

# Download benchmarks
huggingface-cli download --repo-type dataset --resume-download likaixin/ScreenSpot-Pro --local-dir ./data/ScreenSpot-Pro
huggingface-cli download --repo-type dataset --resume-download ServiceNow/ui-vision --local-dir ./data/ui-vision
huggingface-cli download --repo-type dataset --resume-download OS-Copilot/ScreenSpot-v2 --local-dir ./data/ScreenSpot-v2
huggingface-cli download --repo-type dataset --resume-download OpenGVLab/MMBench-GUI --local-dir ./data/MMBench-GUI
huggingface-cli download --repo-type dataset --resume-download vaundys/I2E-Bench --local-dir ./data/I2E-Bench
```

After downloading, some datasets require unzipping compressed image files.

```bash
# Unzip images for ScreenSpot-v2
unzip ./data/ScreenSpot-v2/screenspotv2_image.zip -d ./data/ScreenSpot-v2/

# Unzip images for MMBench-GUI
unzip ./data/MMBench-GUI/MMBench-GUI-OfflineImages.zip -d ./data/MMBench-GUI/
```

### 5. Running the Evaluation

To run the evaluation, use the `eval/eval.py` script. You must specify the path to the model, the benchmark name, and the tensor parallel size.

Here is an example command to evaluate the `InfiGUI-G1-3B` model on the `screenspot-pro` benchmark using 4 GPUs:

```bash
python eval/eval.py \
    ./models/InfiGUI-G1-3B \
    --benchmark screenspot-pro \
    --tensor-parallel 4
```

- **`model_path`**: The first positional argument specifies the path to the downloaded model directory (e.g., `./models/InfiGUI-G1-3B`).
- **`--benchmark`**: Specifies the benchmark to evaluate. Available options include `screenspot-pro`, `screenspot-v2`, `ui-vision`, `mmbench-gui`, and `i2e-bench`.
- **`--tensor-parallel`**: Sets the tensor parallelism size, which should typically match the number of available GPUs.

Evaluation results, including detailed logs and performance metrics, will be saved to the `./output/{model_name}/{benchmark}/` directory.

## 📚 Citation Information

If you find this work useful, citations to the following papers are welcome:

```bibtex
@misc{liu2025infiguig1advancingguigrounding,
      title={InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization}, 
      author={Yuhang Liu and Zeyu Liu and Shuanghe Zhu and Pengxiang Li and Congkai Xie and Jiasheng Wang and Xueyu Hu and Xiaotian Han and Jianbo Yuan and Xinyao Wang and Shengyu Zhang and Hongxia Yang and Fei Wu},
      year={2025},
      eprint={2508.05731},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.05731}, 
}
```

```bibtex
@article{liu2025infigui,
  title={InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners},
  author={Liu, Yuhang and Li, Pengxiang and Xie, Congkai and Hu, Xavier and Han, Xiaotian and Zhang, Shengyu and Yang, Hongxia and Wu, Fei},
  journal={arXiv preprint arXiv:2504.14239},
  year={2025}
}
```

```bibtex
@article{liu2025infiguiagent,
  title={InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection},
  author={Liu, Yuhang and Li, Pengxiang and Wei, Zishu and Xie, Congkai and Hu, Xueyu and Xu, Xinchen and Zhang, Shengyu and Han, Xiaotian and Yang, Hongxia and Wu, Fei},
  journal={arXiv preprint arXiv:2501.04575},
  year={2025}
}
```

## 🙏 Acknowledgements

We would like to express our gratitude for the following open-source projects: [VERL](https://github.com/volcengine/verl), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) and [vLLM](https://github.com/vllm-project/vllm).
