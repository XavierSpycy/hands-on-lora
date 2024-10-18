# Qwen2 -> Named Entity Recognition

## Basic Information
### Setup Details
- Accelerator: NVIDIA RTX 4090D $\times$ 1
- Platform: Linux
- Internet: Enabled

### Model and Resources
- LLM: [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
- Dataset: [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
- Utils: [transformers](https://github.com/huggingface/transformers) | [trl](https://github.com/huggingface/trl)

## Additional Information
> [!IMPORTANT] 
> Key modules are implemented in the [`qwen2ner`](qwen2ner) module. For more technical details, please refer to the module.

- Download the dataset from Kaggle to the `dataset` folder.

- Construct the .csv format dataset.

```bash
python3 construct_text_data.py
```

- Train the model.

```bash
./train.sh
```

- Inference on a single text.

```bash
python3 inference.py \
    --model_name_or_path MODEL_NAME_OR_PATH
```

- 中文博客: [知乎](https://zhuanlan.zhihu.com/p/982156163) ｜ [CSDN](https://blog.csdn.net/NJ_Xavier/article/details/143064590)