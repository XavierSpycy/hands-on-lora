# Llama3 -> Cross-Linguistic Adaptation

## Basic Information
### Setup Details
- Accelerator: NVIDIA RTX 4090D $\times$ 1
- Platform: Linux
- Internet: Enabled

### Model and Resources
- LLM: [Llama3-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct)
- Dataset: `alpaca_zh`,`alpaca_gpt4_zh`,`oaast_sft_zh`
- Utils: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Merged model weights: [XavierSpycy/Meta-Llama-3-8B-Instruct-zh-10k](https://huggingface.co/XavierSpycy/Meta-Llama-3-8B-Instruct-zh-10k)

## Additional Information
> [!IMPORTANT] 
> Deployment-related updates will not be posted here. For detailed deployment updates, please refer to our repository: [llama-ops](https://github.com/XavierSpycy/llama-ops).

- How to use:
    - **Install dependencies** such as torch, transformers, modelscope, etc.

    - **Prepare**: 
        ```bash
        $ source ./prepare.sh
        ```
    
    - Execute Lora training:
        ```bash
        $ source ./train.sh
        ```

        This step takes several hours. Please be patient as the outcomes are well worth the wait.
     
    - Merge the trained adapter with Llama3:
        ```bash
        $ source ./merge.sh
        ```

- Performance comparison

    - Before LoRa:
        ```bash
        $ python3 inference.py
        ```

        Q: 你好，你是谁？

        A: 😊 Ni Hao! I'm a helpful assistant, nice to meet you! I'm here to assist you with any questions, tasks, or topics you'd like to discuss. I'm a language model trained to understand and respond to human language, so feel free to ask me anything! 💬
    
    - After LoRa:
        ```bash
        $ python3 inference.py --model_dir Meta-Llama-3-8B-Instruct-zh-10k
        ```

        Q: 你好，你是谁？

        A: 你好！我是一个人工智能助手，我的名字叫做AI助手。

- `llama.cpp`: Quantization

    - Prepare:

        ```bash
        $ source ./quantize_prepare.sh
        ```
    
    - Quantize:

        ```bash
        $ source ./quantize.sh
        ```
    
    - Test:

        ```bash
        $ source ./quantize_test.sh
        ```

        Terminate the process using `Ctrl` or `Control` plus `C`.

- `llama.cpp`: Deployment
    - Deploy
        - Method 1: Command line

            ```bash
            $ source ./deploy_cli.sh
            ```

            Simlarly, kill the process using `Ctrl` or `Control` plus `C`.

        - Method 2: Docker (Untested)

            ```bash
            $ source ./deploy_docker.sh
            ```
    - Test

        ```bash
        $ source ./deploy_test.sh
        ```
