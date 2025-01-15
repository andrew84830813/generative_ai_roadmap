# generative_ai_roadmap
Generative AI Roadmap

## Slide 2: Agenda & Objectives
**Agenda:**
1. Introduction to Generative AI  
2. What is a Generative AI Application?  
3. Fundamentals of Machine Learning & Neural Networks (TensorFlow/Keras)  
4. Transformers & "Attention Is All You Need"  
5. Popular Tools in Generative AI  
6. Prompt Engineering Techniques  
7. Experimentation & Data Preprocessing with MLflow  
8. Software Development: GitHub and Modular Code  
9. Fine-Tuning, Transfer Learning, LoRA  
10. Retrieval-Augmented Generation (RAG) & Vector Databases  
11. LLM Integration & Deployment (Plotly, Model Serving)  
12. Summary & Q&A  

**Objectives:**
- Understand generative AI basics and its relevance to cybersecurity.
- Learn techniques from ML fundamentals to deployment.
- Gain confidence to begin AI projects using modern frameworks.

## Slide 5: Fundamentals of Machine Learning & Neural Networks (TensorFlow/Keras)
**Content:**
- **Machine Learning Fundamentals:**
  - **Data Preparation:** Emphasize data cleaning, normalization, splitting into train/validation/test sets.
  - **Model Training:** Use loss functions, gradient descent, optimization (Adam, SGD).
  - **Evaluation Metrics:** Accuracy, precision, recall, F1-score; importance in model evaluation.
  - **Overfitting & Underfitting:** Techniques like cross-validation, regularization, dropout.
- **Neural Networks with TensorFlow/Keras:**
  - Structure: Layers of neurons with activation functions.
  - Building deep learning models to learn complex patterns from data.
  
**Code Example (TensorFlow/Keras):**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple feed-forward neural network
model = models.Sequential([
    layers.Dense(50, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
