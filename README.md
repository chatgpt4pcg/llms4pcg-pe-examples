# Prompt Engineering Examples for LLMs4PCG
This repository contains examples of prompt engineering for [LLMs4PCG competition](https://chatgpt4pcg.github.io/2025-llms4pcg). We provide examples of an implementation of the prompt engineering techniques utilizing our Python package, [`llms4pcg-python`](https://github.com/chatgpt4pcg/llms4pcg-python). We implemented several prompt engineering techniques, including:

1. Zero-shot (single-turn and multi-turn versions) prompting (`zero_shot` and `zero_shot_multi_turn` folders).
2. Null-shot prompting (`null_shot` folder).
3. Few-shot prompting (`few_shot` folder).
4. Zero-shot chain-of-thought prompting (`zero_shot_cot` folder).
5. Null-shot chain-of-thought prompting (`null_shot_cot` folder).
6. Tree-of-thought prompting (`tot` folder).

We recommend participants start by taking a look at the `zero_shot` and `zero_shot_multi_turn` examples to understand
the basic implementation of zero-shot prompting. Participants who interest only in optimizing the prompt are welcome to
take a look at the `prompts` folder in each example to see the prompt used in the example and modify it to their needs.
Participants who interest in implementing
more elaborate prompt engineering technique may interest in taking a look at the implementation of the tree-of-thought
prompting in the `tot`
folder.

It is important to note that the examples provided in this repository are not exhaustive and are intended to serve as a
starting point for the participants to experiment with prompt engineering techniques. We encourage the participants to
explore and experiment with different prompt engineering techniques to improve the performance of their approach.
Furthermore, prompts provided in each example are not optimal and can be further improved by the participants.

## Installation

We recommend participants create a virtual environment to run the examples. You can create a virtual environment through
conda or venv. For example, to create a virtual environment using conda, you can run the following command:

```bash
conda create -n llms4pcg-pe-examples python=3.12
```

Once you have created the virtual environment, you can activate the virtual environment by running the following
command:

```bash
conda activate llms4pcg-pe-examples
```

To run the examples, you need to install the required packages. You can install the required packages by running the
following command:

```bash
pip install -r requirements.txt
```

## Running the examples

Navigate to your desired example folder. For example, to run the zero-shot example, you can navigate to the `zero_shot` folder by running the following command:

```bash
cd zero_shot
```

Then, you can run the examples by running the following command:

```bash
python zero_shot.py
```

Please note that you can change the name of the prompt engineering technique to run the other examples. For example, to
run the zero-shot multi-turn example, you can navigate to the `zero_shot_multi_turn` folder from the root directory and run it using the following commands:

```bash
cd zero_shot_multi_turn
python zero_shot_multi_turn.py
```

Please note that the examples provided in this repository are set to generate only one trial for each `A`, `B`, and `C` character. You can
modify the `run_evaluation` function in the examples to generate more trials for different characters.

For example, to generate 10 trials for each `A`, `B`, and `C` character, you can modify the `run_evaluation` function in the `zero_shot.py`
file as follows:

```diff
run_evaluation(team_name=team_name, fn=ZeroShotPrompting,
                model_name=model_name,
                local_model_base_url=local_model_base_url,
-                num_trials=1,
+                num_trials=10,
                characters=["A", "B", "C"])
```

## Contributing

We welcome contributions to this repository to expand the examples and improve the prompt engineering techniques.
