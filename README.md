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
```

## 🚀 Nutzung

App starten

```bash
streamlit run app.py
```

Die App öffnet sich im Browser unter: http://localhost:8501


## 🛠 Trainingsdatensatz generieren (für Fine-Tuning)

```bash
python training.py --input input.json --output training.jsonl
```

Dieses Skript erstellt ein JSONL-Dokument im OpenAI-Fine-Tuning-Format, basierend auf vorhandenen Beschreibungstexten und den zugehörigen Metadaten. Ideal zur Erstellung domänenspezifischer Klassifizierungsmodelle.

## 🌐Beispiel-Vokabular-URLs

https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/educationalContext/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/intendedEndUserRole/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/learningResourceType/index.json

Diese Vokabulare folgen dem SKOS-Standard.

📄 Lizenz

Apache 2.0








