## Continuous Batching & Dynamic Scheduling
**By [Marvin Mboya](https://www.linkedin.com/in/marvin-mboya)** | _Featuring State-of-the-Art LFM2-350M_<br/>

<a href="https://drive.google.com/file/d/1sxAdjaOxrBGpwOsA19MemthMmc3dNxi4/view?usp=sharing">Technical Article Documentation</a><br/>
Large Language Models (LLMs) are large **autoregressive** models that, given a prompt, predict the next tokens (words or sub-words) until the end token. Through sequential stochastic decoding, a prompt response is generated.

> **Definition:** _Autoregressive_ means previous time-step observations are used to predict the current time-step observation. In LLMs, as more tokens are predicted, they are continuously appended to the input to predict even more tokens.

The pioneering paper was published by Google in 2017, and the first network architecture, the **Transformer**, was revealed. However, the first "LLM moment" was by OpenAI in 2018, named GPT-1. Since then, Frontier Labs have trained LLMs on billions and billions of text data. Scaling laws for Transformer-based models show performance improving with model size, data size, and compute. Really powerful open-source models have thus been released such as **DeepSeek, Llama, GPT-OSS, and Mistral**. However, their sizes limit usability in low-memory and low-compute CPU devices. Companies like NVIDIA thus focused on smaller specialized LLMs, **Small Language Models (SLMs)**, which are powerful in agentic systems. This pioneers the working of powerful LLMs for low-memory and low-compute devices. This article takes one such SLM, the **Liquid Foundational Model (350M)**, and builds the model graph and the CPU inference pipeline, dynamically scheduling token generation.
### Implementation & Results
By building the graph from scratch in PyTorch and combining **Conv and KV caching**, **dynamic scheduling**, and **ragged batching**, the model achieves:
-   **Prefill:** ~1.2 tokens/second
-   **Decode:** 45 tokens/second
-   **Overall:** Over 16X batched inference for five prompts.
## References & Links
### Research Papers
-   **[arXiv:2511.23404]** [LFM2 Technical Report](https://arxiv.org/abs/2511.23404) – _Amini et al. 2025_
-   **[arXiv:1706.03762]** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) – _Vaswani et al. 2017_
-   **[arXiv:2001.08361]** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) – _Kaplan et al. 2020_
-   **[arXiv:2506.02153]** [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) – _Belcak et al. 2025_
### Acknowledgments
Many thanks to LiquidAI open-source powerful LFM models, and Hugging Face team for the article on [Continuous Batching](https://huggingface.co/blog/continuous_batching) written by Reboul, Zucker, and Georges, which motivated the writing of this article!