
<h1 align="center">
  <br>
  <a href="#"><img src="https://i.postimg.cc/Nfm19CDJ/banner.jpg" alt="ClauseX" width="1500"></a>
  <br>
  <br>
  ClauseX: Automated Contract Validation Tool
  <br>
</h1>

<!---h2 align="center">Business Contract Validation -To Classify Content within the Contract Clauses and Determine Deviations from Templates and highlight them.</h2--->

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#objective">Objective</a> •
  <a href="#deliverables">Deliverables</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#models">Models</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#user-interfaces">User Interfaces</a> •
  <a href="#conclusion">Conclusion</a>
</p>


ClauseX is a powerful contract validation tool leveraging the BERT model. It automates the process of contract review by classifying clauses and highlighting deviations, ensuring faster and more efficient validation. This project aims to reduce legal risks and ensure compliance with legal standards.

## Acknowledgements
We would like to thank our mentors and contributors for their invaluable support and guidance. Special thanks to Siju Swamy, for providing us with the expertise and resources necessary to complete this project. We also extend our gratitude to anyone who has supported or helped us in any way.

## Introduction

In the complex world of business contracts, ensuring compliance and minimizing legal risks is paramount. ClauseX addresses this need by providing a streamlined solution for contract validation. Using advanced natural language processing capabilities of the BERT model, ClauseX efficiently processes complex legal documents, classifies clauses, and detects deviations.

## Objective

The objective of this project is to build a robust tool that automates the validation of business contracts, ensuring every word counts and reducing the risk of non-compliance.

## Deliverables

We have addressed the following:

Data Collection: Various types of business contracts were created. Data Annotation: Key entities and clauses were labeled to create a structured dataset for training. Data Preparation: The data was prepared for training. Model Training: The BERT model was fine-tuned on the annotated dataset to learn the specific legal language and contextual nuances. Model Evaluation: The performance of the model was assessed based on accuracy, precision, recall and F1 score. Model Selection: The best-performing model was selected based on evaluation metrics. Making Predictions: The selected model was used to classify new contracts and highlight deviations.


## Dataset

The **Business-Contract-Dataset-Intel-Training-Program-2024** is a comprehensive dataset designed for training and evaluating models on contract analysis. It consists of 27 contract files organized into 5 distinct folders based on contract type. This dataset is ideal for those looking to train machine learning models for tasks such as clause classification, deviation detection, and contract parsing.
The dataset is organized into the following folders, each containing contracts specific to its type:

- **Employment**: 5 contracts
- **Joint**: 5 contracts
- **Partnership**: 7 contracts
- **Purchase**: 4 contracts
- **Sales**: 5 contracts

[Dataset Folder](https://github.com/meghaarajeev/Business-Contract-Dataset-Intel-Training--Program-2024.git)

## Models

We have addressed the following:

Data Collection: Various types of business contracts were collected. Data Annotation: Key entities and clauses were labeled to create a structured dataset for training. Data Preparation: The data was prepared for training. Model Training: The BERT model was fine-tuned on the annotated dataset to learn the specific legal language and contextual nuances. Model Evaluation: The performance of the model was assessed based on accuracy, precision, recall and F1 score. Model Selection: The best-performing model was selected based on evaluation metrics. Making Predictions: The selected model was used to classify new contracts and highlight deviations.

## Evaluation

The performance of the trained model was evaluated using various metrics such as accuracy, precision, recall and F1 score.

## User Interfaces

ClauseX provides an intuitive user interface for uploading and validating contracts. Below are the main interfaces of ClauseX:

- **Home Interface**: The starting page where users can access the main functionalities.

  <img src="https://i.ibb.co/JKdnjLm/landingpage-ezgif-com-video-to-gif-converter.gif" width="750" height="400" alt="Home">

- **Validation Interface**: Users can upload contracts in PDF format for validation.
  
  <img src="https://i.ibb.co/cQf3DRb/image-2024-07-14-171559631.png" width="750" height="400" alt="Validation">

- **Result Interface**: Displays the validation results, including detected clauses and deviations.

  <img src="https://i.ibb.co/XZKG8Pb/image-2024-07-14-171052117.png" width="750" height="400" alt="Result Page">


## Conclusion

ClauseX significantly reduces the time and effort required for contract validation, ensuring compliance with legal standards and minimizing risks. Its advanced NLP capabilities make it a valuable tool for businesses of all sizes.

For more information, please refer to the detailed documentation and code provided in this repository.

## Team Members

- [Megha Rajeev](https://github.com/meghaarajeev)
- [Aiswarya Lakshmi](https://github.com/Lakshmiaishwarya01)
- [Bestin Biju](https://github.com/bestin06)
- [Neethu Benny](https://github.com/Neet-hu)
- [Shane Sam Manappallil](https://github.com/shanesamofficial)

<!---```bash
# Clone this repository
$ git clone https://github.com/amitmerchant1990/electron-markdownify

# Go into the repository
$ cd electron-markdownify

# Install dependencies
$ npm install

# Run the app
$ npm start
``` --->

