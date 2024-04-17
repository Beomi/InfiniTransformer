# InfiniTransformer

Unofficial PyTorch/🤗Transformers(+Gemma) implementation of Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

- Paper Link: https://arxiv.org/abs/2404.07143

## Two types of Implemenation for Infini-Attention

**Part I. Infini Attention in Attention-Layer only**

- Overrides modeling python file only, especially Attention layer only.
- Minimal edit, fully compatible with HF(Trainer, etc)
- Memory usage is ~eq with SDPA(default) attention
  - can train Gemma-2B with 8192 seq len(128*64) on 2x H100 80G (with Adafactor Optimizer + Gradient Checkpointing)

**Part II. Infini Attention in Model-wise, Trainer-wise**

- Overrides modeling and config python files.
- Full edit, Not compatible with basic HF trainer.
- Need custom training code
- Memory usage is **much lower** than SDPA(default) attention
  - can train Gemma-2B with 32768 seq len(2048*16) on 2x H100 80G (with AdamW optimizer, No gradient checkpointing)

## How to use Part I. Infini Attention in Attention-Layer only

### 1. Clone this repository

```bash
git clone https://github.com/Beomi/InfiniTransformer
```

### 2. Install dependencies

> We need to install the latest version(`b109257f4f`) of 🤗Transformers from the source code.

```bash
pip install -r requirements.txt
pip install -e git+https://github.com/huggingface/transformers.git@b109257f4f#egg=transformers
```

### 3. Remove original `modeling_gemma.py`, make a symbolic link with new `modeling_gemma.py`

```bash
python test_basic.infini.py
```

### 4. Run the example(Inference, simple forward/backward test)

```bash
python test_basic.py
```

### 5. Train with your data

```bash
python test_train.small.gemma.py
```

<img width="808" alt="image" src="https://github.com/Beomi/InfiniTransformer/assets/11323660/c3cb7b1e-531c-4652-a5de-fcf36b1c03bc">

Example code used wikitext-2-raw-v1 from https://huggingface.co/datasets/wikitext

Here's the test wandb log here -> https://api.wandb.ai/links/beomi2/1rsqrkfn

### 6. Inference

```bash
python test_basic.trained.py
```

**Sample Generation w/ 1-epoch Trained Model on WikiText2**

Input:

> This work introduces an efficient method to scale Transformer-based

Output1:

> models for denoising , denoising denoising , and deep denoising of images of the U2 EPK model , using a coefficient that is a function of the depth of the image resolution . The paper experiments with image denoising by Turbo @-@ based filtering , denoising by generative adversarial networks , and video denoising by denoising each of the three elements of the video ( color of the pixels / frames ) . The results are considered fair . The video is not discussed . The paper is not considering an actual application in an industrial context ,  line is probably a 1 . It is built in the Nohmi…

Output2:

> vision models across platforms using a custom architecture optimized for both vision ( 3D / 2D ) and vision and language . In other words , a single model can run on different types of devices , a feature that is critical for the development of general @-@ purpose and large-scale AI ( see also : The One @-@ Model @-@ for @-@ All @-@ Things @-@ AI Problem ) . The model is the first to reach a global scale ( 200 GPU + ) on a single GPU using the Transformer and its variants . The model can run at the end of 1967 . He had his family relocated to a house in a nearby neighborhood , where they lived for five years , before returning to their primary residence in St. Petersburg . Later comments of 1968 made by his fellow musician Bruce Hornsby made it clear that he had gone through a lot , both personally and professionally .

## How to use Part II. Infini Attention in Model-wise, Trainer-wise.

### 1. Clone this repository

```bash
git clone https://github.com/Beomi/InfiniTransformer
```

### 2. Install dependencies

> We need to install the latest version(`b109257f4f`) of 🤗Transformers from the source code.

```bash
pip install -r requirements.txt
pip install -e git+https://github.com/huggingface/transformers.git@b109257f4f#egg=transformers
# or just pip install transformers
```

### 3. Run the example(Inference, simple forward/backward test)

```bash
python test_basic.infini.py
```

### 4. Train with your data

```bash
./train.gemma.infini.noclm.sh
```
