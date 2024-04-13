# InfiniTransformer

> WIP: This repository is under development.

Unofficial PyTorch/🤗Transformers(+Gemma) implementation of Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

- Paper Link: https://arxiv.org/abs/2404.07143

## How to use

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
rm ./src/transformers/src/transformers/models/gemma/modeling_gemma.py
ln -s $(pwd)/modeling_gemma.py ./src/transformers/src/transformers/models/gemma/modeling_gemma.py
```

### 4. Run the example

```bash
python test.py
```
