---
title: Projects
---

### FastServe
[Repo](https://github.com/aniketmaurya/fastserve)

Machine Learning Serving focused on GenAI with simplicity as the top priority.
Dynamic batching powered by pure Python implementation.

**To serve Image-to-Text API using SSD-1B:**

```python
from fastserve.models import FastServeSSD

serve = FastServeSSD(device="cuda", batch_size=4, timeout=1)
serve.run_server()
```

---

### Muse (Stable Diffusion Deployment)
[Repo](https://github.com/Lightning-Universe/stable-diffusion-deploy)

Open source, stable-diffusion production server to show how to deploy diffusion models in a real production environment with: load-balancing, gpu-inference, performance-testing, micro-services orchestration and more. All handled easily with the Lightning Apps framework.


![Muse](https://user-images.githubusercontent.com/3640001/195984024-788255e7-d01b-4522-9655-2a3ba56e80aa.png)

---

### Discord LLM Bot
[Repo](https://github.com/aniketmaurya/discord-llm-bot)

Retrieval Augmented Genration (RAG) powered Discord Bot that works seamlessly on CPU. Powered by LanceDB and Llama.cpp.
This Discord bot is designed to helps answer questions based on a knowledge base (vector db).


![](https://raw.githubusercontent.com/aniketmaurya/discord-llm-bot/main/assets/discord-bot.png){height="420"}

---

### LLM Inference
[Repo](https://github.com/aniketmaurya/llm-inference)

Large Language Model (LLM) Inference API and Chatbot 🦙

![](https://github.com/aniketmaurya/llm-inference/raw/main/assets/llm-inference-min.png){height="420"}

**Build and run LLM Chatbot under 7 GB GPU memory in 5 lines of code.**

```python
from llm_chain import LitGPTConversationChain, LitGPTLLM
from llm_inference import prepare_weights

path = str(prepare_weights("meta-llama/Llama-2-7b-chat-hf"))
llm = LitGPTLLM(checkpoint_dir=path, quantize="bnb.nf4")  # 7GB GPU memory
bot = LitGPTConversationChain.from_llm(llm=llm, prompt=llama2_prompt_template)

print(bot.send("hi, what is the capital of France?"))
```
---

### Gradsflow
[Repo](https://github.com/gradsflow/gradsflow)

An open-source AutoML Library based on PyTorch

![model training image](https://ik.imagekit.io/gradsflow/docs/gf/gradsflow-model-training_B1HZpLFRv8.png)

---


### Chitra
[Repo](https://github.com/aniketmaurya/chitra)

{{< video https://youtu.be/r1VZeKamprE
    title="Automatic Building Docker Image for Any Machine Learning or Deep Learning Model 🐳"
    width="426" height="240"
>}}

A multi-functional library for full-stack Deep Learning. Simplifies Model Building, API development, and Model Deployment.

![](https://ik.imagekit.io/aniket/chitra/chitra-arch_Vw9AdA4aC.svg){height="420"}
