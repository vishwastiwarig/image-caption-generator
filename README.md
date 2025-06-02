🖼️ Image Caption Generator
An AI-powered image caption generator that automatically describes images using a CNN-LSTM deep learning model. This project combines computer vision and natural language processing to generate human-like descriptions of images.
🌟 Features

Automatic Image Captioning: Generate descriptive captions for any image
Deep Learning Architecture: CNN (Xception) + LSTM model
Kaggle Ready: Optimized for Kaggle notebooks with GPU support
Interactive Testing: Easy-to-use functions for testing your own images
Visual Results: Display images with generated captions
Model Evaluation: Built-in performance metrics and testing

🏗️ Model Architecture
Image Input (299x299x3)
        ↓
   Xception CNN (Pre-trained)
        ↓
   Feature Vector (2048)
        ↓
   Dense Layer (256) → Dropout
        ↓
Text Input (max_length) → Embedding (256) → LSTM (256)
        ↓
   Combine Features (Add)
        ↓
   Dense (256) → Output (vocab_size)
📊 Dataset

Dataset: Flickr8k (8,091 images with 40,455 captions)
Images: Various scenes, people, animals, objects
Captions: 5 human-written descriptions per image
Vocabulary: ~7,500 unique words after cleaning
Max Caption Length: 20-25 words

🚀 Quick Start (Kaggle)
1. Setup Dataset
python# Add Flickr8k dataset to your Kaggle notebook:
# 1. Click "+ Add Data" in right panel
# 2. Search for "flickr8k" 
# 3. Add "adityajn105/flickr8k" dataset
2. Enable GPU
python# In Kaggle Settings:
# Accelerator → GPU T4 x2
3. Run the Code
python# Copy and run the main script
# Training will take ~45 minutes with GPU
💻 Usage
Generate Caption for Any Image
python# Test with dataset image
test_any_dataset_image("1000268201_693b08cb0e")

# Test your own uploaded image
test_my_image("/kaggle/input/your-image/photo.jpg")

# Test multiple random images
quick_test()
Model Information
python# Print model architecture
print(model.summary())



# Performance evaluation
evaluate_model()
Interactive Functions
pythoncaption_image_improved("image_id")    # Caption specific image
test_multiple_images(5)               # Test 5 random images
run_interactive_demo()                # Interactive demo
📁 Project Structure
image-caption-generator/
├── main_script.py              # Main training script
├── improved_model.py           # Enhanced model version
├── models/                     # Saved models
│   ├── model_epoch_1.h5
│   ├── model_epoch_2.h5
│   └── best_model.h5
├── data/                       # Dataset files
│   ├── features.pkl            # Extracted image features
│   ├── tokenizer.pkl           # Text tokenizer
│   └── descriptions.txt        # Cleaned captions
└── outputs/                    # Generated results
    ├── model_architecture.png
    └── sample_results/
🔧 Technical Details
Model Specifications

CNN Backbone: Xception (pre-trained on ImageNet)
RNN: LSTM with 256 hidden units
Embedding: 256-dimensional word embeddings
Optimizer: Adam (learning_rate=0.0005)
Loss Function: Categorical Crossentropy
Batch Size: 32-64
Epochs: 5-8

Data Processing

Image Preprocessing: Resize to 299×299, normalize to [-1,1]
Text Preprocessing: Lowercase, remove punctuation, filter short words
Sequence Padding: Post-padding for GPU compatibility
Vocabulary: Top ~7,500 most frequent words


📈 Performance Metrics
Model Performance

Training Accuracy: 70-80%
Validation Accuracy: ~60-70%
Average Caption Length: 6-8 words
Word Overlap with Ground Truth: ~40-50%


Local Environment
bashpip install tensorflow>=2.8.0
pip install pillow matplotlib pandas numpy
pip install kaggle  # For dataset download
🎯 Usage Examples
Basic Usage
python# Load trained model
model = load_model('/kaggle/working/best_model.h5')
tokenizer = pickle.load(open('/kaggle/working/tokenizer.pkl', 'rb'))


🐛 Troubleshooting
Common Issues
1. CUDA/GPU Errors
python# Solution: Disable cuDNN
LSTM(256, use_cudnn=False)
# Or force CPU usage
tf.config.set_visible_devices([], 'GPU')
2. Memory Errors
python# Solution: Reduce batch size
batch_size = 16  # Instead of 64
max_images = 1000  # Instead of 2000
3. Poor Caption Quality
python# Solutions:
# - Train for more epochs (8-10)
# - Use larger dataset (full 8k images)
# - Improve text preprocessing
# - Use temperature sampling
4. Dataset Not Found
python# Check paths:
print(os.listdir("/kaggle/input/"))
# Make sure you added the correct Flickr8k dataset

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Flickr8k Dataset: University of Illinois
Xception Model: Google Research
TensorFlow/Keras: Google
Kaggle: For free GPU compute



📚 References

Vinyals, O., et al. "Show and Tell: A Neural Image Caption Generator" (2015)
Chollet, F. "Xception: Deep Learning with Depthwise Separable Convolutions" (2017)
Young, P., et al. "From image descriptions to visual denotations" (2014)
