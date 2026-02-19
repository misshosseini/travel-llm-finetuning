# ✈️ Travel Assistant LLM – Fine-Tuning Pipeline

## 📌 Project Overview
This project implements a complete fine-tuning pipeline for **:contentReference[oaicite:0]{index=0}** using Supervised Fine-Tuning (SFT) and LoRA (Low-Rank Adaptation) to build an intelligent travel assistant chatbot.

The fine-tuned model can:
- Understand travel-related queries  
- Detect user intent  
- Generate accurate and helpful responses  
- Handle multi-intent travel scenarios  

---

## 📚 Dataset

This project uses the **:contentReference[oaicite:1]{index=1}** from **:contentReference[oaicite:2]{index=2}**  
(`bitext/Bitext-travel-llm-chatbot-training-dataset`)

### Dataset Details
- ~31,000 Q&A pairs  
- 33 travel-related intents (e.g., flight booking, baggage info, trip planning)  
- Fields:
  - `instruction` → User query  
  - `intent` → Travel intent label  
  - `response` → Target answer  

### Data Preparation
- Loaded using the `datasets` library  
- Grouped by intent  
- Balanced via equal sampling per intent (e.g., 50 samples for demo)  
- Formatted into structured conversation strings:

