import json
import os
import argparse
import time
from typing import List, Dict, Any
from openai import OpenAI

def extract_text_from_json(item: Dict[str, Any]) -> str:
    """Extract text content from a JSON item to be analyzed."""
    text = ""
    
    # Add title if available
    if "title" in item:
        text += f"Title: {item['title']}\n\n"
    
    # Add description if available
    if "properties" in item and "cclom:general_description" in item["properties"]:
        text += f"Description: {item['properties']['cclom:general_description'][0]}\n\n"
    
    # Add keywords if available
    if "properties" in item and "cclom:general_keyword" in item["properties"]:
        keywords = ", ".join(item["properties"]["cclom:general_keyword"])
        text += f"Keywords: {keywords}\n\n"
    
    return text.strip()

def fetch_vocabularies(urls):
    """Fetch and parse vocabulary data from URLs."""
    vocabularies = []
    for url in urls.strip().split('\n'):
        if url.strip():
            try:
                import requests
                response = requests.get(url.strip())
                if response.status_code == 200:
                    vocab_data = response.json()
                    vocab_name = url.split('/')[-1].replace('.json', '')
                    
                    # Extract property name from title or URL
                    property_name = vocab_data.get('title', {}).get('de', '')
                    if not property_name:
                        property_name = vocab_data.get('title', {}).get('en', '')
                    if not property_name:
                        property_name = vocab_name
                    
                    # Extract property and possible values with labels
                    values = []
                    if 'hasTopConcept' in vocab_data:
                        for item in vocab_data.get('hasTopConcept', []):
                            value_id = item.get('id', '')
                            value_label = item.get('prefLabel', {}).get('de', '')
                            if not value_label:
                                value_label = item.get('prefLabel', {}).get('en', '')
                            
                            if value_id and value_label:
                                values.append({"id": value_id, "label": value_label})
                            
                            # Include narrower concepts if available
                            for narrower in item.get('narrower', []):
                                narrower_id = narrower.get('id', '')
                                narrower_label = narrower.get('prefLabel', {}).get('de', '')
                                if not narrower_label:
                                    narrower_label = narrower.get('prefLabel', {}).get('en', '')
                                
                                if narrower_id and narrower_label:
                                    values.append({"id": narrower_id, "label": narrower_label})
                    
                    if values:
                        vocabularies.append({
                            "name": vocab_name,
                            "property": property_name,
                            "values": values
                        })
                        print(f"Successfully loaded vocabulary: {property_name} ({len(values)} values)")
                    else:
                        print(f"No values found in vocabulary: {vocab_name}")
                else:
                    print(f"Failed to fetch vocabulary from {url}: Status code {response.status_code}")
            except Exception as e:
                print(f"Error processing vocabulary from {url}: {str(e)}")
    return vocabularies

def extract_actual_metadata(content_item, vocabularies):
    """Extract the actual metadata values from a content item JSON based on vocabulary URIs."""
    actual_metadata = {}
    
    if "properties" not in content_item:
        return actual_metadata
    
    properties = content_item.get("properties", {})
    
    # Scan all properties to find those containing vocabulary URIs
    for vocab in vocabularies:
        # Get the base URI without the numeric ID at the end
        vocab_base_uri = None
        if vocab["values"] and len(vocab["values"]) > 0:
            # Extract the base URI from the first value's ID
            sample_id = vocab["values"][0]["id"]
            # Find the last occurrence of '/' and remove everything after it
            last_slash_index = sample_id.rfind('/')
            if last_slash_index > 0:
                vocab_base_uri = sample_id[:last_slash_index]
        
        if not vocab_base_uri:
            continue
        
        # Look for properties containing values with this base URI
        matches = []
        for prop_key, prop_values in properties.items():
            if not isinstance(prop_values, list):
                continue
                
            for value in prop_values:
                if not isinstance(value, str):
                    continue
                    
                if value.startswith(vocab_base_uri):
                    # This property contains values from this vocabulary
                    # Find the label for this ID in the vocabulary
                    for vocab_value in vocab["values"]:
                        if value == vocab_value["id"]:
                            matches.append({
                                "id": value,
                                "value": vocab_value["label"]
                            })
                            break
        
        if matches:
            actual_metadata[vocab["property"]] = matches
    
    return actual_metadata

def create_training_data(input_file, output_file, vocab_urls, api_key=None):
    """
    Create a JSONL training file for OpenAI fine-tuning from the input JSON file.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSONL file
        vocab_urls: URLs of vocabularies to use
        api_key: OpenAI API key (optional)
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load vocabularies
    print("Loading vocabularies...")
    vocabularies = fetch_vocabularies(vocab_urls)
    
    if not vocabularies:
        print("No vocabularies loaded. Exiting.")
        return
    
    print(f"Processing {len(data)} items...")
    training_examples = []
    
    for i, item in enumerate(data):
        if i % 10 == 0:
            print(f"Processing item {i+1}/{len(data)}")
            
        # Extract text content
        text = extract_text_from_json(item)
        if not text:
            print(f"Skipping item {i+1}: No text content found")
            continue
        
        # Extract actual metadata
        actual_metadata = extract_actual_metadata(item, vocabularies)
        if not actual_metadata:
            print(f"Skipping item {i+1}: No metadata found")
            continue
        
        # Create the completion format
        completion = {
            "analysis": []
        }
        
        # Add metadata for each property
        for prop_name, matches in actual_metadata.items():
            if matches:
                completion["analysis"].append({
                    "property": prop_name,
                    "matches": [
                        {
                            "value": match["value"], 
                            "id": match["id"], 
                            "confidence": 90, 
                            "explanation": f"Das Material ist mit {match['value']} assoziiert."
                        } for match in matches
                    ]
                })
        
        # Skip if no analysis was created
        if not completion["analysis"]:
            print(f"Skipping item {i+1}: No analysis created")
            continue
        
        # Create the training example
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "Du bist ein Assistent, der Bildungsmaterialien analysiert und passende Metadaten zuordnet. Analysiere den Text und bestimme die passenden Werte für Schulfach, Bildungsstufe, Zielgruppe und Inhaltstyp."
                },
                {
                    "role": "user",
                    "content": f"Analysiere den folgenden Text und bestimme die passenden Metadaten:\n\n{text}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(completion, ensure_ascii=False)
                }
            ]
        }
        
        training_examples.append(training_example)
    
    # Write the training examples to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Created {len(training_examples)} training examples in {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create OpenAI fine-tuning data from JSON")
    parser.add_argument("--input", default="result_wlo_staging_redaktionsbuffet_12032025_sampled_lrt.json", 
                        help="Input JSON file path")
    parser.add_argument("--output", default="trainingsdatensatz.jsonl", 
                        help="Output JSONL file path")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI API key (optional)")
    
    args = parser.parse_args()
    
    # Default vocabulary URLs
    default_vocabs = """https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/educationalContext/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/intendedEndUserRole/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/learningResourceType/index.json"""
    
    create_training_data(args.input, args.output, default_vocabs, args.api_key)

if __name__ == "__main__":
    main()