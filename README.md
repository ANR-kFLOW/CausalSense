
# Refined Causality Extraction from Text

This repository contains the data and scripts used for refined causality extraction from text.

## Data

The [`data` directory](https://github.com/ANR-kFLOW/Relation_extraction/tree/main/data) contains the **CausalSense dataset** used in our experiments.

## Pipeline

In this directory, it is contained the **pipeline**, the **API** and the **demo**.
Please refer to the [paper](https://www.eurecom.fr/publication/8673) and to the README inside the folder.

## End-to-End Models

The `end_to_end_models` directory includes models used for training, specifically:

- **REBEL**
- **RoBERTa_end_to_end**

### Training Instructions

- **REBEL**:  
  - Train the REBEL model using:  

    ```bash
    python end_to_end_models/REBEL/train.py
    ```

  - Transform data into the format required for REBEL-based event relation extraction (ERE) using:  

    ```bash
    python end_to_end_models/REBEL/Data_transform.py
    ```

    *Ensure to adjust the script as needed for your dataset.*

- **RoBERTa_end_to_end**:  
  - Train the RoBERTa model for end-to-end event relation extraction using:  

    ```bash
    python end_to_end_models/RoBERTa_end_to_end/end_to_end_train.py
    ```

## Separate Task Training

For training each subtask in refined causality extraction, use the following scripts:

- **Relation Detection**:  

  ```bash
  python separate_tasks/Relation_detection.py
  ```

- **Relation Classification**:  

  ```bash
  python separate_tasks/relation_classification.py
  ```

- **Event Extraction**:  

  ```bash
  python separate_tasks/EE.py
  ```

# Citation

For citing this work, please use ([bibtex](./rebboud2026causalsense.bib)):

> Youssra Rebboud, Pasquale Lisena, and Raphael Troncy. 2026. CausalSense: Leveraging common sense knowledge and LLMs for joint event extraction and relation classification. In LREC 2026, International Conference on Language Resources and Evaluation, 11-16 May 2026, Palma, Mallorca, Spain. https://www.eurecom.fr/publication/8673

Other related publications:

-  Gustavo Flores Miguel, Youssra Rebboud, Pasquale Lisena, Raphäel Troncy. 2025.
**Streamlining Event Relation Extraction: A Pipeline Leveraging Pretrained and Large Language Models for Inference.**
In: *EKAW 2024, 24th International Conference on Knowledge Engineering and Knowledge Management*, Poster and Demo Track, CEUR-WS, Nov 2024, Amsterdam, Netherlands.
https://ceur-ws.org/Vol-3967/PD_paper_184.pdf

- Youssra Rebboud, Pasquale Lisena, and Raphael Troncy. 2023. Prompt-based Data Augmentation for Semantically-Precise Event Relation Classification. In Semantic Methods for Events and Stories workshop (SEMMES), CEUR-WS, 29 May 2023, Hersonissos, Greece. https://ceur-ws.org/Vol-3443/ESWC_2023_SEMMES_Data_Augmentation.pdf
