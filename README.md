# 🧠 SkoHub ZeroShot Classification

Dieses Tool ist eine Streamlit-App zur **Zero-Shot-Klassifizierung** von Texten mit Hilfe von **kontrollierten Vokabularen** (z. B. aus [skohub-vocabs](https://vocabs.openeduhub.de)).  
Es nutzt die OpenAI API, um Texte automatisch mit passenden Metadaten zu annotieren – etwa Schulfach, Bildungsstufe, Zielgruppe oder Inhaltstyp.

## 🔍 Hauptfunktionen

- **Zero-Shot-Klassifizierung** von beliebigem Beschreibungstext basierend auf Vokabular-URIs.
- **Vokabular-Import** über JSON-LD-Dateien (z. B. von `vocabs.openeduhub.de`).
- **Einzeltext-Analyse** mit Confidence-Werten und Erklärungen.
- **Batch-Analyse** von JSON-Dateien mit Lerninhalten und automatischer Bewertung der Klassifizierungsgüte (Precision, Recall, F1).
- **Generierung von Trainingsdaten** für OpenAI-Fine-Tuning im JSONL-Format.

## 🧰 Voraussetzungen

- Python 3.9+
- OpenAI API-Key

## 📦 Installation

```bash
pip install -r requirements.txt
