# ChatEV
This is a very simple implementation of **utilizing large language models (e.g., Llama-3.2-1B-Instruct) for time-series forecasting** in the scenarios of electric vehicle charging. If it is helpful to your research, please cite our papers:

>Haohao Qu, Han Li, Linlin You, Rui Zhu, Jinyue Yan, Paolo Santi, Carlo Ratti, Chau Yuen. (2024) ChatEV: Predicting electric vehicle charging demand as natural language processing. Transportation Research Part D: Transport and Environment. [Paper in TRD](https://doi.org/10.1016/j.trd.2024.104470)


### Environments
We need five major packages, namely torch, pandas, numpy, transformers, and argparse. You can install these useful wheels by:

```shell
pip install -r requirements.txt
```

### Meta-Llama hf_token
To get access to Meta-Llama models, we need to apply a "hf_token" key through https://huggingface.co/settings/tokens

Then replace input a correct "hf_token" in Line 99 of the "utils.py" file.
```shell
hf_token = "Your_HF_TOKEN"
```

ps: Or you can download a local model through https://www.llama.com/llama-downloads/

### Implementation
To conduct a simple implementation (inference only), we can run the "simple.py" file.
```shell
python simple.py
```

### Questions
* If you uncounter a problem of slow downloading, you can set a mirror source at the command terminal:
```shell
export HF_ENDPOINT=https://hf-mirror.com
```
