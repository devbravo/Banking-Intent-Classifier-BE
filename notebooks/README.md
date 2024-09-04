### Banking Intent Classification - Colab Notebook

**Overview**

This notebook focuses on building and training a deep learning model to classify customer intents in the banking sector. It covers steps from data analysis and preprocessing to model training and evaluation.

**Notebook Outline**

	1.	Text Augmentation:
	•	We apply various text augmentation techniques such as synonym replacement and random word swapping.
	2.	Data Exploration and Analysis:
	•	Visualize the distribution of sequence lengths and word frequencies using histograms and word clouds.
	3.	Preprocessing:
	•	Tokenization, vocabulary creation, numericalization, and padding are applied to prepare the text data for model input.
	4.	Model Training:
	•	A bidirectional LSTM model is built and trained using PyTorch. The model is trained with Cross-Entropy Loss and an AdamW optimizer.
	5.	Evaluation:
	•	Evaluate the model using accuracy and classification reports on the test dataset.

  **Key Findings**
  - The model achieved a accury of 85% 
  - To Improve model accuracy and accuracy on certain classes more training data is needed
