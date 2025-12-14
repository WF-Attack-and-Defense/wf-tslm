<div align="center">
  <h2><b> WF-TSLM: Time-Series LLMs for Website Fingerprinting on Tor Traffic </b></h2>
</div>



>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:


## Updates/News:

🚩 **News** (Dec. 2025): WF-TSLM has be released.


## Requirements
Use python 3.12 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](), then place the downloaded contents under `./dataset`

## Quick Demos
1. Download datasets and place them under `./dataset`
2. Tune the model. We provide five experiment scripts for demonstration purpose under the folder `./scripts`. For example, you can evaluate on AWF or NetCLR datasets by:

```bash
bash ./scripts/TimeLLM_Traffic.sh
```

## Detailed usage

Please refer to ```run_main.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.


## Further Reading

As one of the earliest works exploring the intersection of large language models and time series, we sincerely thank the open-source community for supporting our research. While we do not plan to make major updates to the main Time-LLM codebase, we still welcome **constructive pull requests** to help maintain and improve it.

🌟 Please check out our team’s latest research projects listed below. 

1, [**Time-LLM: Time series forecasting by reprogramming large language models**](https://arxiv.org/abs/2310.01728), *arXiv* 2024.

**Authors**: Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
