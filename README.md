![Logo](https://github.com/AKGanesh/Conv_chatbot/blob/main/chatbot.png)

# Conversational Chatbots (WIP)
### V1- Seq2Seq Model with LSTM for Text Generation
### V2- Transformer-Based Seq2Seq Model
### V3- Using pre-trained model
---------

### V1- Seq2Seq Model with LSTM for Text Generation
This project implements a Seq2Seq model with an LSTM architecture for text generation. The model is trained on a dataset of Shakespearean text to predict the next word in a sequence.

Dataset: https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

    Text Preprocessing:
    - Loads Shakespearean text data.
    - Tokenizes the text into sequences of integers.
    - Creates input-output pairs, considering context windows
    - Pads sequences to a fixed length.

    Model Architecture:
    - Encoder-decoder architecture with LSTMs.
    - Encoder embeds the input sequence and captures context
    - Decoder predicts the next word based on the encoder output and its own hidden state.

    Training:
    - Trains the model to minimize the sparse categorical crossentropy loss.
    - Uses Adam optimizer and accuracy metric.
    - Implements early stopping to prevent overfitting.
    - Saves the best model checkpoint based on validation loss.

    Limitations:
    - This model uses a basic LSTM architecture.
    - More advanced architectures like Transformer models could potentially achieve better performance.

    Possible Applications:
    - Text generation tasks like creative writing, dialogue generation, or machine translation.
    
    Future Work:
    - Experiment with different network architectures (e.g., Transformer)
    - Explore beam search and temperature sampling for more diverse outputs.
    - Train the model on specific datasets for tailored text generation tasks.


### V2- Transformer-Based Seq2Seq Model
This version implements a Seq2Seq model based on the Transformer architecture, designed for tasks like text generation, translation, and summarization. The model leverages self-attention mechanisms to capture long-range dependencies within sequences.


    Encoder:
    - Input Embedding: Converts input tokens into dense vectors.
    - Positional Encoding: Adds positional information to the embeddings.
    - Transformer Encoder Blocks: Multiple layers, each consisting of:
        Self-Attention: Captures dependencies within the input sequence.
        Feed-Forward Neural Network: Applies non-linear transformations.

    Decoder:
    - Input Embedding: Converts input tokens (target sequence) into dense vectors.
    - Positional Encoding: Adds positional information to the embeddings.
    - Transformer Decoder Blocks: Multiple layers, each consisting of:
        Masked Self-Attention: Prevents the model from attending to future tokens.
        Encoder-Decoder Attention: Allows the decoder to attend to relevant parts of the encoder's output.
        Feed-Forward Neural Network: Applies non-linear transformations.

    Output:
    - A dense layer with softmax activation to predict the next token.


### V3- Transformer-Based Seq2Seq Model
This project demonstrates using a pre-trained T5 model for text generation tasks. T5 is a powerful Transformer-based model from Google AI, trained on a massive dataset for various text-to-text tasks.

    Pre-trained Model Leverage:
    - Utilizes the pre-trained T5 model "t5-base" for text generation.
    - Avoids time-consuming training from scratch.
    - Benefits from the model's learned knowledge of text relationships
    
    Text preparation:
    - Assumes pre-existing data with paired input and target sequences.
    - Decodes numerical sequences to human-readable text for clarity.
    - Tokenizes and pads sequences for compatibility with the T5 model.

    Fine Tuning:
    - Fine-tunes the pre-trained T5 model on your specific data for improved performance.
    - Requires providing both input and target sequences during training.
    - Employs the 'adam' optimizer and categorical crossentropy loss.

    Limitations:
    - Relies on pre-trained model capabilities, potentially requiring adjustments for specific tasks.
    - Fine-tuning might not achieve optimal performance compared to training from scratch on a smaller but highly relevant dataset.

    Future Work:
    - Expriment with hugging face models like BERT
    - Evaluate the model's performance using appropriate metrics (e.g., BLEU score for machine translation
    - Implement beam search or temperature sampling for more diverse generation outputs.


## Libraries

**Language:** Python

**Packages:** Pandas, Numpy, Matplotlib, Tensorflow, Keras, Transformers

## Evaluation and Results
-- WIP --

## FAQ

#### What is Embedding?
Embedding in the context of natural language processing (NLP) is a technique used to represent words or phrases as dense vectors in a continuous space. These vectors, often referred to as embeddings, capture the semantic meaning and relationships between words.

Key concepts:

- Dense vectors: Embeddings are represented as fixed-length numerical vectors, unlike one-hot encoding which results in sparse vectors.
- Continuous space: The vectors exist in a continuous space, allowing for smooth transitions between similar words.
- Semantic relationships: Embeddings capture the semantic relationships between words, meaning similar words will have similar embeddings.

#### What are common embedding techniques?
- Word2Vec: A popular technique that learns word embeddings by predicting surrounding words in a text corpus.
- GloVe: A technique that learns word embeddings by factoring a co-occurrence matrix.
- FastText: An extension of Word2Vec that learns embeddings for subwords, making it more effective for handling out-of-vocabulary words.
- BERT: BERT (Bidirectional Encoder Representations from Transformers) is a more recent technique that learns contextualized word embeddings. It uses a transformer architecture to model the relationship between words in a sentence.

#### TextVectorization vs Tokenizer
- TextVectorization is a layer in TensorFlow that combines tokenization and text-to-sequence conversion. It provides a more streamlined approach to preparing text data for machine learning models. It automatically standardizes text data, including lowercasing, punctuation removal, and whitespace normalization.
In many cases, TextVectorization is a more convenient and efficient choice. However, for more complex scenarios or custom tokenization requirements, using Tokenizer might be necessary.

#### What is a sequence
{
    "this": 1,
    "is": 2,
    "a": 3,
    "sample": 4,
    "text": 5,
    "for": 6,
    "tokenization": 7,
    "another": 8
}

Then, the sequence for the text "This is another sample text." would be: [1, 2, 8, 4, 5]

#### LSTM vs Transformer vs Pretrained
While LSTM-based Seq2Seq models were once the state-of-the-art, Transformer models have surpassed them in performance for many NLP tasks.

Pre-trained Transformer models like BERT provide a strong foundation and can be fine-tuned for specific tasks with minimal effort. However, the choice of model depends on the specific task, available resources, and desired performance.

#### What is attention
In the context of neural networks, especially in sequence-to-sequence models like Transformers, attention is a mechanism that allows the model to focus on specific parts of the input sequence when processing the output sequence. This enables the model to capture long-range dependencies and weigh the importance of different input tokens.

By understanding the attention mechanism, you can better appreciate the power of Transformer models and how they are able to achieve state-of-the-art performance on various NLP tasks.

https://arxiv.org/abs/1706.03762 (Attention is all you need)

## Acknowledgements
- https://www.tensorflow.org/text/tutorials/nmt_with_attention
- https://www.tensorflow.org/text/tutorials/transformer
- https://huggingface.co/docs/transformers/en/model_doc/t5


## Contact

For any queries, please send an email (id on my github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)