# ATELIER2_DL

# Lab Synthesis

## Part 1: CNN Classifier
**Objective:** Implement a Convolutional Neural Network (CNN) architecture using PyTorch to classify the MNIST dataset.
**Key Learnings:**
- Understanding the architecture of CNNs including convolutional layers, pooling layers, and fully connected layers.
- Setting hyperparameters such as kernel size, padding, stride, and learning rate.
- Utilizing PyTorch's GPU capabilities for faster training.
**Metrics to Evaluate:**
- Accuracy: Percentage of correctly classified images.
- F1 Score: Harmonic mean of precision and recall.
- Loss: Measure of the model's performance during training.
- Training Time: Time taken to train the model.

## Part 2: Faster R-CNN
**Objective:** Apply Faster R-CNN, originally designed for object detection, to the MNIST dataset for classification.
**Key Learnings:**
- Adapting complex architectures for different tasks.
- Understanding the challenges and limitations of repurposing models for different tasks.
**Metrics to Evaluate:**
- Similar metrics as Part 1 (Accuracy, F1 Score, Loss, Training Time).

## Part 3: Model Comparison
**Objective:** Compare the performance of the CNN Classifier and Faster R-CNN.
**Key Learnings:**
- Assessing the effectiveness of different architectures for the same task.
- Understanding trade-offs between model complexity and performance.
**Metrics for Comparison:**
- Accuracy, F1 Score, Loss, and Training Time.

## Part 4: Transfer Learning with VGG16 and AlexNet
**Objective:** Fine-tune pre-trained models (VGG16 and AlexNet) on the MNIST dataset and compare the results with CNN and Faster R-CNN.
**Key Learnings:**
- Leveraging pre-trained models for transfer learning.
- Understanding the benefits of transfer learning in different scenarios.
**Metrics for Comparison:**
- Accuracy, F1 Score, Loss, and Training Time.

q4 conclusion 
# Conclusion

Based on the experimental results obtained from fine-tuning the models on the MNIST dataset, the following conclusions can be drawn:

1. **CNN Classifier Performance**:
   - Achieved an accuracy ranging from 98.52% to 99.09% over 5 epochs.
   - Shows consistent improvement in accuracy with each epoch, indicating effective learning.

2. **Fine-tuned CNN Performance**:
   - Achieved an accuracy ranging from 98.46% to 99.09% over 5 epochs.
   - Shows comparable performance to the original CNN classifier, indicating that fine-tuning did not significantly impact performance.

3. **Fine-tuned VGG16 Performance**:
   - Achieved an accuracy ranging from 98.43% to 99.13% over 5 epochs.
   - Shows competitive performance compared to the CNN classifier and fine-tuned CNN, indicating the effectiveness of transfer learning with a pre-trained model.

Overall, all models demonstrate high accuracy in classifying the MNIST dataset. Fine-tuning VGG16 on the MNIST dataset shows promise, suggesting that pre-trained models can be effectively utilized for transfer learning tasks. However, further analysis and experimentation may be needed to determine the most optimal model for the specific classification task at hand.

## Part 2: Vision Transformer (ViT)
**Objective:** Implement a Vision Transformer (ViT) model architecture from scratch and perform classification on the MNIST dataset.
**Key Learnings:**
- Understanding the concept of Vision Transformers and their architecture.
- Implementing Vision Transformers using PyTorch.
- Adapting Vision Transformers for image classification tasks.
**Metrics to Evaluate:**
- Accuracy: Percentage of correctly classified images.
- F1 Score: Harmonic mean of precision and recall.
- Loss: Measure of the model's performance during training.
- Training Time: Time taken to train the model.




 
## Conclusion
- The lab provides hands-on experience with building and comparing different neural network architectures for image classification tasks.
- Participants gain insights into model design, hyperparameter tuning, and performance evaluation.
- Understanding the trade-offs between model complexity, training time, and performance is crucial for selecting the most suitable architecture for a given task.


