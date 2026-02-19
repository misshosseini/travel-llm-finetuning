Travel LLM Finetuning: Building an Intelligent Travel Chatbot ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)OverviewThis project demonstrates fine-tuning the TinyLlama-1.1B-Chat model using Supervised Fine-Tuning (SFT) and LoRA for creating an intelligent travel assistant chatbot. The model is adapted to handle travel-related queries, intent detection, and responses based on a balanced subset of the Bitext travel dataset.Key features:Domain Adaptation: Fine-tuned for travel intents like flight booking, baggage info, and trip planning.
Efficient Training: Uses LoRA for parameter-efficient fine-tuning to reduce computational requirements.
Dataset Balancing: Samples from 33 intents to ensure balanced training data.

This repository serves as a practical example of LLM fine-tuning for conversational AI, suitable for portfolios or further extensions like RAG integration.DatasetSource: Bitext Travel LLM Chatbot Dataset (~31k Q&A pairs across 33 intents).
Preprocessing: Balanced sampling (e.g., 50 records for quick training) and formatting into conversation strings: "Query: {instruction}\nIntent: {intent}\nResponse: {response}".

InstallationClone the repository:

git clone https://github.com/yourusername/travel-llm-finetuning.git
cd travel-llm-finetuning

Install dependencies:

pip install transformers trl peft datasets torch

Requires Python 3.8+ and access to a GPU for efficient training (optional but recommended).UsagePrepare Data: The script loads and balances the dataset automatically.
Fine-Tune the Model:

python air.py

Adjust max_steps or num_train_epochs in SFTConfig for longer training.

Generate Responses:
After training, the script generates a sample response. Example output for "Query: I'm trying to book a flight":

Intent: book_flight
Response: To book a flight, visit our website at {{WEBSITE_URL}} or use the app {{APP_NAME}}...

For custom inference:python

inputs = tokenizer.encode("Query: What's the best time to visit Paris?", return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

Results and EvaluationTrained on 50 samples for demonstration (extend for better performance).
Post-fine-tuning, the model generates context-aware travel responses.
Metrics: Monitor loss during training (logged in /tmp output dir).

Future improvements: Integrate evaluation metrics (e.g., BLEU/ROUGE) or deploy as a Gradio app.ContributingContributions are welcome! Please fork the repo and submit a pull request. For major changes, open an issue first.LicenseThis project is licensed under the MIT License - see the LICENSE file for details.AcknowledgmentsHugging Face for Transformers, TRL, and PEFT libraries.
Bitext for the open travel dataset.

