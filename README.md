# TourGPT: Elevate Your Adventure with Cutting-Edge AI Tour Guidance! 🌟

Welcome to the future of exploration! **TourGPT** is not just a tour guide – it's your ultimate travel companion, powered by advanced language modeling, deep learning, and real-time web interaction. Say goodbye to static tours — and hello to dynamic, AI-powered experiences!

---

## 🚀 Features:

- 🤖 **AI-Driven Tour Guidance**  
  TourGPT is built on a custom GPT-like language model that generates human-like, destination-aware travel content in real time.

- ⚙️ **Built with PyTorch**  
  The core model is implemented in PyTorch using the Transformer architecture, enabling efficient text generation and strong language understanding.

- 🌐 **Real-Time Interaction via WebSocket**  
  Integrated with Flask and Flask-SocketIO, allowing interactive responses through a web interface. Just type and receive AI-powered insights instantly!

- 📊 **From Scratch Training**  
  The model is trained on a custom dataset of travel-related text using character-level tokenization and transformer-based attention blocks.

---

## 📚 Dataset

TourGPT is trained on a **curated dataset** (`datt.txt`) sourced from diverse travel and tourism content across the internet, focusing on:
- Landmarks and local attractions
- Cultural facts and trivia
- Travel tips and regional insights

This ensures TourGPT can serve rich, destination-specific responses.

---

## 🧠 How It Works

- **Language Model**: A character-level GPT architecture trained from scratch, using self-attention, multi-head transformers, and autoregressive decoding.

- **Training**: The model is trained on ~90% of the dataset, evaluated on the rest, and saved as `model.pth` for fast inference.

- **Prediction**: Given an input phrase (like “Tell me about Paris”), TourGPT continues the text intelligently using the trained model.

- **Web Integration**: A Flask app receives user queries via WebSocket, runs the model's prediction logic, and emits the result back to the browser.

---

## ⚙️ Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Prayash-Das/GPTTour.git
   cd TourGPT
Install Python dependencies:
pip install torch flask flask-socketio
(Optional) Train the model:
If you'd like to retrain:

from tourgpt import trainModel
trainModel()
Run the app:
python app.py
Then open your browser at: http://localhost:6005
📡 Tech Stack

PyTorch
Flask + Flask-SocketIO
HTML5 frontend (index.html)
GPT-style Transformer (built from scratch)
Character-level tokenizer and sampler
🧪 Model Architecture Highlights

6 Transformer Blocks
384-dimensional embeddings
Multi-head Attention (6 heads)
Causal Masking
Cross-Entropy Loss for next-token prediction
👨‍💻 Contributors

Prayash Das
📜 License

MIT License – feel free to use and build upon TourGPT.

✨ Get Ready to Explore Like Never Before

Whether you're planning a vacation or simulating an AI-driven travel assistant, TourGPT is ready to guide, inform, and inspire.
