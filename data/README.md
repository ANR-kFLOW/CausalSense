# CausalSense Dataset

**CausalSense** is a large-scale dataset designed for **fine-grained causal event relation extraction**. It supports the identification of causal relationships between events expressed in natural language, going beyond simple causation by including more nuanced relations.

The dataset contains sentences annotated with five relation types:

* **Cause** – an event directly causes another event
* **Enable** – an event facilitates or makes another event possible
* **Prevent** – an event stops or blocks another event
* **Intend** – an event occurs with the intention of causing another event
* **No-relation** – no causal relation between the events

To address the scarcity and imbalance of fine-grained causal annotations, CausalSense combines data from **multiple sources**:

* existing manually annotated datasets
* **news corpora**
* **commonsense knowledge bases (ATOMIC)**
* **synthetic data generated with LLMs**

This combination produces a dataset of **over 500k sentences**, enabling the training of models capable of recognizing subtle causal semantics.

The dataset is used both for **training large-scale models** and for evaluating **joint event extraction and causal relation classification systems**.

---

## Dataset Statistics

| Category             | Dataset                              |       Total |      Cause |     Enable |    Prevent |      Intend |     No-rel. |
| -------------------- | ------------------------------------ | ----------: | ---------: | ---------: | ---------: | ----------: | ----------: |
| News Data            | Original Data ([Rebboud et al., 2022](https://github.com/ANR-kFLOW/event-relation-classification/tree/main/data)) |         663 |        268 |        100 |         81 |          42 |         172 |
| News Data            | Synthetic Data (GPT-3.5)             |       1,228 |          0 |        350 |        419 |         459 |           0 |
| News Data            | Causal News Corpus (CNC)             |       3,316 |      1,710 |          0 |          0 |           0 |       1,606 |
| Common Sense         | [ATOMIC](https://huggingface.co/datasets/allenai/atomic)                               |     315,173 |     82,242 |          0 |          0 |     146,588 |      86,943 |
| Common Sense         | Synthetic CommonSense                |     205,884 |          0 |     65,485 |     53,456 |           0 |      86,943 |
| **Total**            |                                      | **526,264** | **84,321** | **66,025** | **54,067** | **147,189** | **175,664** |
| **Combined dataset** |                                      |   **6,792** |  **3,520** |    **814** |    **948** |     **944** |     **566** |
| **Test dataset**     |                                      |     **632** |    **351** |     **89** |     **52** |      **40** |     **100** |
|                      | including **A VeriTeC**              |     **216** |    **133** |     **46** |     **26** |      **11** |       **0** |

---

For a more detailed description of the dataset, please refer to the [paper](https://www.eurecom.fr/publication/8673) ([bibtex](./rebboud2026causalsense.bib)):

> Youssra Rebboud, Pasquale Lisena, and Raphael Troncy. 2026. CausalSense: Leveraging common sense knowledge and LLMs for joint event extraction and relation classification. In LREC 2026, International Conference on Language Resources and Evaluation, 11-16 May 2026, Palma, Mallorca, Spain
