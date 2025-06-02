# 🖼️ Image Caption Generator

• **AI-powered image caption generator** that automatically describes images using CNN-LSTM deep learning model
• **Combines computer vision and natural language processing** to generate human-like descriptions of images

## 🌟 Features

• **Automatic Image Captioning** - Generate descriptive captions for any image
• **Deep Learning Architecture** - CNN (Xception) + LSTM model
• **Kaggle Ready** - Optimized for Kaggle notebooks with GPU support
• **Interactive Testing** - Easy-to-use functions for testing your own images
• **Visual Results** - Display images with generated captions
• **Model Evaluation** - Built-in performance metrics and testing

## 🏗️ Model Architecture

```
• Image Input (299x299x3)
        ↓
• Xception CNN (Pre-trained)
        ↓
• Feature Vector (2048)
        ↓
• Dense Layer (256) → Dropout
        ↓
• Text Input (max_length) → Embedding (256) → LSTM (256)
        ↓
• Combine Features (Add)
        ↓
• Dense (256) → Output (vocab_size)
```

![Model Architecture](https://github.com/user-attachments/assets/78de3a55-dd75-48fd-ab13-00d6ca32ed80)

## 📊 Dataset

• **Dataset**: Flickr8k (8,091 images with 40,455 captions)
• **Images**: Various scenes, people, animals, objects  
• **Captions**: 5 human-written descriptions per image
• **Vocabulary**: ~7,500 unique words after cleaning
• **Max Caption Length**: 20-25 words



## 🔧 Technical Specifications

### Model Components
• **CNN Backbone**: Xception (pre-trained on ImageNet)
• **RNN**: LSTM with 256 hidden units
• **Embedding**: 256-dimensional word embeddings
• **Optimizer**: Adam (learning_rate=0.0005)
• **Loss Function**: Categorical Crossentropy
• **Batch Size**: 32-64
• **Epochs**: 5-8

### Data Processing
• **Image Preprocessing**: Resize to 299×299, normalize to [-1,1]
• **Text Preprocessing**: Lowercase, remove punctuation, filter short words
• **Sequence Padding**: Post-padding for GPU compatibility
• **Vocabulary**: Top ~7,500 most frequent words

## 📈 Performance Metrics

### Model Performance
• **Training Accuracy**: 70-80%
• **Validation Accuracy**: ~60-70%
• **Average Caption Length**: 6-8 words
• **Word Overlap with Ground Truth**: ~40-50%

### Load and Use Model
```python
• # Load trained model
• model = load_model('/kaggle/working/best_model.h5')
• tokenizer = pickle.load(open('/kaggle/working/tokenizer.pkl', 'rb'))

• # Generate caption
• caption = generate_caption(model, tokenizer, image_features, max_length)
• print(f"Generated: {caption}")
```

### Batch Processing
```python
• # Process multiple images
• image_ids = ["1000268201_693b08cb0e", "1001773457_577c3a7d70"]
• for img_id in image_ids:
•     caption = test_any_dataset_image(img_id)
•     print(f"{img_id}: {caption}")
```

## 🎯 Performance Tips

• **Use GPU**: Enable T4 x2 in Kaggle for 10x faster training
• **Larger Dataset**: Use all 8,091 images for better results  
• **More Epochs**: Train for 8-10 epochs for optimal performance
• **Temperature Sampling**: Use temperature=0.7 for diverse captions



## 📄 License

• This project is licensed under the **MIT License**

## 🙏 Acknowledgments

• **Flickr8k Dataset**: University of Illinois
• **Xception Model**: Google Research  
• **TensorFlow/Keras**: Google
• **Kaggle**: For free GPU compute

