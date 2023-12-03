# Personalized Daily Arxiv Papers 12/03/2023
Total relevant papers: 4

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations](#link0)
**Authors:** Raphael Tang, Xinyu Zhang, Jimmy Lin, Ferhan Ture

1. [X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning](#link1)
**Authors:** Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, Juan Carlos Niebles

2. [CritiqueLLM: Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation](#link2)
**Authors:** Pei Ke, Bosi Wen, Zhuoer Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, Jie Tang, Minlie Huang

3. [IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions](#link3)
**Authors:** Zhebin Zhang, Xinyu Zhang, Yuanhang Ren, Saijiang Shi, Meng Han, Yongkang Wu, Ruofei Lai, Zhao Cao

---
## 0. [What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations](https://arxiv.org/abs/2311.18812) <a id="link0"></a>
**ArXiv ID:** 2311.18812
**Authors:** Raphael Tang, Xinyu Zhang, Jimmy Lin, Ferhan Ture

**Abstract:** Do large language models (LLMs) exhibit sociodemographic biases, even when they decline to respond? To bypass their refusal to "speak," we study this research question by probing contextualized embeddings and exploring whether this bias is encoded in its latent representations. We propose a logistic Bradley-Terry probe which predicts word pair preferences of LLMs from the words' hidden vectors. We first validate our probe on three pair preference tasks and thirteen LLMs, where we outperform the word embedding association test (WEAT), a standard approach in testing for implicit association, by a relative 27% in error rate. We also find that word pair preferences are best represented in the middle layers. Next, we transfer probes trained on harmless tasks (e.g., pick the larger number) to controversial ones (compare ethnicities) to examine biases in nationality, politics, religion, and gender. We observe substantial bias for all target classes: for instance, the Mistral model implicitly prefers Europe to Africa, Christianity to Judaism, and left-wing to right-wing politics, despite declining to answer. This suggests that instruction fine-tuning does not necessarily debias contextualized embeddings. Our codebase is at https://github.com/castorini/biasprobe.


---

## 1. [X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning](https://arxiv.org/abs/2311.18799) <a id="link1"></a>
**ArXiv ID:** 2311.18799
**Authors:** Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, Juan Carlos Niebles

**Abstract:** Vision-language pre-training and instruction tuning have demonstrated general-purpose capabilities in 2D visual reasoning tasks by aligning visual encoders with state-of-the-art large language models (LLMs). In this paper, we introduce a simple, yet effective, cross-modality framework built atop frozen LLMs that allows the integration of various modalities without extensive modality-specific customization. To facilitate instruction-modality fine-tuning, we collect high-quality instruction tuning data in an automatic and scalable manner, composed of 24K QA samples for audio and 250K QA samples for 3D. Leveraging instruction-aware representations, our model performs comparably with leading-edge counterparts without the need of extensive modality-specific pre-training or customization. Furthermore, our approach demonstrates cross-modal reasoning abilities across two or more input modalities, despite each modality projection being trained individually. To study the model's cross-modal abilities, we contribute a novel Discriminative Cross-modal Reasoning (DisCRn) evaluation task, comprising 9K audio-video QA samples and 28K image-3D QA samples that require the model to reason discriminatively across disparate input modalities.


---

## 2. [CritiqueLLM: Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation](https://arxiv.org/abs/2311.18702) <a id="link2"></a>
**ArXiv ID:** 2311.18702
**Authors:** Pei Ke, Bosi Wen, Zhuoer Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, Jie Tang, Minlie Huang

**Abstract:** Since the natural language processing (NLP) community started to make large language models (LLMs), such as GPT-4, act as a critic to evaluate the quality of generated texts, most of them only train a critique generation model of a specific scale on specific datasets. We argue that a comprehensive investigation on the key factor of LLM-based evaluation models, such as scaling properties, is lacking, so that it is still inconclusive whether these models have potential to replace GPT-4's evaluation in practical scenarios. In this paper, we propose a new critique generation model called CritiqueLLM, which includes a dialogue-based prompting method for high-quality referenced / reference-free evaluation data. Experimental results show that our model can achieve comparable evaluation performance to GPT-4 especially in system-level correlations, and even outperform GPT-4 in 3 out of 8 tasks in a challenging reference-free setting. We conduct detailed analysis to show promising scaling properties of our model in the quality of generated critiques. We also demonstrate that our generated critiques can act as scalable feedback to directly improve the generation quality of LLMs.


---

## 3. [IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions](https://arxiv.org/abs/2311.18397) <a id="link3"></a>
**ArXiv ID:** 2311.18397
**Authors:** Zhebin Zhang, Xinyu Zhang, Yuanhang Ren, Saijiang Shi, Meng Han, Yongkang Wu, Ruofei Lai, Zhao Cao

**Abstract:** Retrieval-Augmented Generation (RAG), by incorporating external knowledge with parametric memory of language models, has become the state-of-the-art architecture for open-domain QA tasks. However, common knowledge bases are inherently constrained by limited coverage and noisy information, making retrieval-based approaches inadequate to answer implicit reasoning questions. In this paper, we propose an Induction-Augmented Generation (IAG) framework that utilizes inductive knowledge along with the retrieved documents for implicit reasoning. We leverage large language models (LLMs) for deriving such knowledge via a novel prompting method based on inductive reasoning patterns. On top of this, we implement two versions of IAG named IAG-GPT and IAG-Student, respectively. IAG-GPT directly utilizes the knowledge generated by GPT-3 for answer prediction, while IAG-Student gets rid of dependencies on GPT service at inference time by incorporating a student inductor model. The inductor is firstly trained via knowledge distillation and further optimized by back-propagating the generator feedback via differentiable beam scores. Experimental results show that IAG outperforms RAG baselines as well as ChatGPT on two Open-Domain QA tasks. Notably, our best models have won the first place in the official leaderboards of CSQA2.0 (since Nov 1, 2022) and StrategyQA (since Jan 8, 2023).


---


---

## Paper selection prompt
 1. New methodological improvements to RLHF or instruction-following which are specific fine-tuning steps that are taken to make language models better at following user instructions across a range of tasks.
    - Relevant: papers that discuss specific methods like RLHF, or instruction-tuning datasets, improving these methods, or analyzing them. Usually these papers will explicitly mention RLHF, instruction-following or instruction-tuning.
    - Not relevant: papers about adaptation to some task. Simply following instructions or inputs are not sufficient.
 2. Shows new powerful test set contamination or membership inference methods for language models. Test set contamination is the phenomenon where a language model observes a benchmark dataset during pretraining.
    - Relevant: test statistics that can detect contamination of benchmarks in language models. statistics that can provide guarantees are more interesting. membership inference methods that are general enough to apply to language models are also relevant.
    - Not relevant: any papers that do not consider language models, or that do not consider test set contamination.
 3. Studies 'scaling laws' in the context of neural networks. Scaling laws refer to the very clear power-law relationship between the size or computational power used to train a model and the performance of that model.
    - Relevant: theoretical or conceptual explanation behind scaling laws for language models.
    - Not relevant: papers that have experiments at different model scales (but do not explicitly fit a scaling law) or papers that mention scaling laws, but the scaling laws are not the central subject of the paper

 In suggesting papers to your friend, remember that he enjoys papers on statistical machine learning, and generative modeling in natural language processing.
 Your friend also likes learning about surprising empirical results in language models, as well as clever statistical tricks.
 He does not want to read papers that are about primarily applications of methods to specific domains.