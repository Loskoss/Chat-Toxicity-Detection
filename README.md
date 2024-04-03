# Real-time Toxicity Detection System 
Real-time toxicity detection system for live chat environments, utilizing machine learning models and integrated into chat bots for moderation.

A machine learning model is trained to detect toxic comments in live chat, allowing for real-time moderation. The model is integrated into chat bots for platforms like Twitch. The project also explores future tasks such as implementing an ASCII spam detector and expanding the model's capabilities using NLP techniques.

## Model Creation
The toxicity classification model is trained using various machine learning algorithms and evaluated using metrics such as F1 score, precision, recall, and ROC AUC score.

### Key Findings
- Best Performer: The SVM model demonstrated the highest overall performance, achieving F1 scores of 0.94 for severe toxic comments and 0.85 for threatening comments.
- Other Models: Logistic Regression and Random Forests also performed well, especially for identifying insulting comments, with F1 scores of 0.94.
- **The trained model achieves an accuracy of 92.3% on the combined dataset. Toxicity probabilities are calculated for incoming messages, and if they exceed a predefined threshold, a response is triggered to maintain chat cleanliness.**
  
## Future Work
1. Implement an ASCII spam detector to identify and filter out spam messages.
2. Explore advanced NLP techniques to enhance the model's performance and handle complex language nuances.

## License
This project is licensed under the MIT License.

