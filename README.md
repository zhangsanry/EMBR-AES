# EMBR-AES
## EMBR-AES: Expert-Mode and Deep Learning Integrated Automatic Essay Scoring System

### Project Introduction

In recent years, various fields and levels of academic competitions have emerged, leading to increased student participation. With the growing number of participating teams, the workload for reviewing papers has risen sharply, creating greater demands on human resources. Therefore, Automatic Essay Scoring (AES) systems have become a necessity, capable of reducing workload and improving the fairness of evaluations.

This paper proposes a new essay scoring framework—**EMBR-AES**, aimed at enhancing the performance of automatic scoring systems by integrating expert models with deep learning methods. The **EMBR-AES** framework captures **shallow features** through universally defined expert patterns and utilizes the **ALBERT** model to extract **deep semantic features**. Finally, it employs a **neural network model** to uncover latent patterns and structures related to essay scoring.

### Features
- **Integration of expert-defined rules with deep learning technology**: Combines traditional rule-based scoring with advanced neural networks.
- **Transparent and interpretable scoring process**: Ensures clear and justifiable results.
- **High scoring speed**: Capable of handling large-scale competition applications efficiently.
- **Good adaptability and accuracy**: Demonstrates strong performance across different datasets.

---

## File Structure

This repository contains several key modules that contribute to the functionality of the **EMBR-AES** framework:

- **`Algorithms/`**  
  Contains implementations of traditional machine learning models for essay scoring, providing benchmark methods for comparison with deep learning-based approaches.

- **`Student/`**  
  Includes the downstream task processing for different **Student models**, responsible for learning features from **embedding vectors** generated by the encoding module.

- **`Encoder/`**  
  Stores the **encoding module**, which corresponds to the **Source Feature Extraction** process in the paper. This module extracts textual features using pre-trained deep learning models.

- **`Expert_Mode/`**  
  Contains models related to the **expert-mode scoring approach**, corresponding to the **Initial Paper Scoring** module in the research framework.

- **`Results/`**  
  Stores all **model execution results**, including intermediate outputs and final evaluation results.

- **`Trainer/`**  
  Contains the training scripts and model files trained on the **TDBSW academic scientific paper dataset**, supporting the fine-tuning of scoring models.

- **`Visualization/`**  
  Includes scripts for **generating visualizations** based on experimental results, particularly used to create figures **5, 6, 7, and 8** in the paper.

- **`Accuracy/`**  
  Provides evaluation scripts for calculating the final accuracy of the models.

- **`Dataset/`**  
  A compressed archive of the **TDBSW dataset**, which serves as the benchmark dataset for training and evaluating the scoring models.

- **`Description/`**  
  Documents the **essay scoring criteria**, outlining the grading framework used for model evaluation.

- **`Each_Grade/`**  
  Stores the detailed scores for different aspects of essays, including **subscores, total scores, and corresponding grade labels**, derived based on the established scoring criteria.

---

## **TDBSW Dataset**
The **TDBSW dataset** originates from the **"Teddy Cup" Data Mining Challenge**. 

The **"Teddy Cup" Data Mining Challenge** is a national-level academic competition aimed at **graduate and undergraduate students**. It is designed to encourage students to actively engage in data mining, enhancing their ability to analyze and solve real-world problems. 

This competition is organized by the **Teddy Cup Data Mining Challenge Committee**, and the competition topics are primarily sourced from **real-world problems provided by enterprises, government institutions, and research organizations**. Participants are required to possess **statistical and data mining knowledge** as well as proficiency in relevant software tools.

For more details about the **Teddy Cup Data Mining Challenge**, please visit the official website:  
[**https://www.tipdmcup.cn/**](https://www.tipdmcup.cn/).  

This website provides the latest competition updates and relevant resources.

---

This README provides an overview of **EMBR-AES**, its core functionalities, and its file structure, ensuring that users can effectively navigate the repository and understand the implementation details. If you have any questions or require further details, please feel free to reach out.
