[English](README.md) | 中文
<h1>动手 LoRa: 用 LoRa 实践微调大模型</h1>
<img src="https://img.shields.io/badge/License-Apache_2.0-lightblue.svg">

<h2 style="font-family:'Comic Sans MS', sans-serif; color: green;"><em>深度学习是一门实验科学。若不熟能，如何生巧？</em></h2>

<h2>引言</h2>
<a href="https://arxiv.org/abs/2106.09685" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2106.09685-b31b1b.svg?style=plastic" alt="arXiv">
</a> : "LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each
layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."
<p>【翻译】: LoRA, 冻结了预训练模型的权重, 并在每一层 Transformer 架构中注入可训练的低秩分解矩阵, 极大地减少了下游任务的可训练参数量。</p>
<img src="img/lora.jpg">
<div align="center" style="font-weight: bold;">
    可训练的 A & B (仅)
</div>

<h2>示例</h2>
<div align="center">
      <table style="text-align: center;">
          <tr>
            <td>大语言模型</td>
            <td>参数量</td>
            <td>任务</td>
            <td>LoRa/QLoRa</td>
            <td>代码</td>
          </tr>
          <tr>
            <td>Gemma-IT</td>
            <td>2B</td>
            <td>文生文</td>
            <td>QLoRa</td>
            <td><a href="examples/gemma_text2text">链接</a></td>
          </tr>
          <tr>
            <td>Qwen 2</td>
            <td>1.5B</td>
            <td>命名实体识别</td>
            <td>QLoRa</td>
            <td><a href="examples/qwen_ner">链接</a></td>
          </tr>
          <tr>
            <td>Llama 3</td>
            <td>8B</td>
            <td>跨语言适应</td>
            <td>LoRa</td>
            <td><a href="examples/llama_chinese">链接</a></td>
          </tr>
      </table>
  </div>

> [!提示]
> 尽管 LoRa 是一项很优雅的技术, 用它微调大模型仍然需要大量工程上的努力。最优性能需要全面的优化。在我们的仓库中，我们提供了基础示例，可视其为起点。达到优异仍有一定的距离。我们鼓励您发挥您的天赋和创造力，实现更出色的结果。