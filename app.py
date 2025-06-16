import streamlit as st
import requests
import json
import os
from openai import OpenAI
import time  # Add this import for sleep functionality

# Set page title
st.set_page_config(page_title="Metadata Checker", layout="wide")

# App title
st.title("Text Metadata Analysis Tool")

# Function to fetch and parse vocabulary data
def fetch_vocabularies(urls):
    vocabularies = []
    for url in urls.strip().split('\n'):
        if url.strip():
            try:
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
                        st.success(f"Successfully loaded vocabulary: {property_name} ({len(values)} values)")
                    else:
                        st.warning(f"No values found in vocabulary: {vocab_name}")
                else:
                    st.error(f"Failed to fetch vocabulary from {url}: Status code {response.status_code}")
            except Exception as e:
                st.error(f"Error processing vocabulary from {url}: {str(e)}")
    return vocabularies

# Function to extract text from JSON content
def extract_text_from_json(content_item):
    """Extract the general description text from a content item JSON."""
    if "properties" in content_item:
        properties = content_item.get("properties", {})
        general_description = properties.get("cclom:general_description", [])
        if general_description and len(general_description) > 0:
            return general_description[0]
    return None

# Function to extract actual metadata from JSON content
# Function to extract actual metadata from JSON content
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

# Function to calculate precision and F1 score
def calculate_metrics(predicted, actual):
    """Calculate precision and F1 score for metadata predictions."""
    metrics = {}
    
    for property_name, actual_matches in actual.items():
        predicted_matches = next((p["matches"] for p in predicted.get("analysis", []) 
                                if p["property"] == property_name), [])
        
        # Extract IDs for comparison
        actual_ids = set(match["id"] for match in actual_matches)
        predicted_ids = set(match["id"] for match in predicted_matches)
        
        # Calculate metrics
        true_positives = len(actual_ids.intersection(predicted_ids))
        false_positives = len(predicted_ids - actual_ids)
        false_negatives = len(actual_ids - predicted_ids)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[property_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    # Calculate overall metrics
    all_precision = sum(m["precision"] for m in metrics.values()) / len(metrics) if metrics else 0
    all_recall = sum(m["recall"] for m in metrics.values()) / len(metrics) if metrics else 0
    all_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics) if metrics else 0
    
    metrics["overall"] = {
        "precision": all_precision,
        "recall": all_recall,
        "f1": all_f1
    }
    
    return metrics

# Function to analyze text with OpenAI
def analyze_text(text, vocabularies, api_key, model="gpt-4o-mini", max_retries=2):
    if not api_key:
        st.error("OpenAI API key is required")
        return None
    
    client = OpenAI(api_key=api_key)
    
    # Build prompt with structured output format
    prompt = f"""Analyze the following text and determine the most appropriate metadata values from the provided vocabularies.
    
Text to analyze:
{text}

Vocabularies:
"""
    
    for vocab in vocabularies:
        prompt += f"\nProperty: {vocab['property']}\n"
        prompt += "Possible values:\n"
        for value in vocab['values']:
            prompt += f"- {value['label']} (ID: {value['id']})\n"
    
    prompt += """\nProvide your analysis as a JSON with the following structure:
{
  "analysis": [
    {
      "property": "Property Name",
      "matches": [
        {
          "value": "Value Label",
          "id": "Value ID",
          "confidence": 85,
          "explanation": "Explanation for this match"
        }
      ]
    }
  ]
}

Include confidence scores (0-100%) for each match and provide a brief explanation for why each value was selected.
"""
    
    # Implement retry logic
    attempts = 0
    while attempts <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model,  # Use the selected model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            attempts += 1
            error_msg = f"Error during analysis (attempt {attempts}/{max_retries+1}): {str(e)}"
            st.error(error_msg)
            
            if attempts <= max_retries:
                st.info(f"Retrying in 2 seconds... (Attempt {attempts+1}/{max_retries+1})")
                time.sleep(2)  # Wait before retrying
            else:
                st.error("All retry attempts failed.")
                return None

# Function to analyze batch of content items
def analyze_batch(content_items, vocabularies, api_key, model="gpt-4o-mini"):
    """Analyze a batch of content items and compare with actual metadata."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Counters for overall metrics
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for i, item in enumerate(content_items):
        # Update progress
        progress = (i + 1) / len(content_items)
        progress_bar.progress(progress)
        status_text.text(f"Processing item {i+1} of {len(content_items)}: {item.get('title', 'Unnamed item')}")
        
        # Extract text to analyze
        text = extract_text_from_json(item)
        if not text:
            results.append({
                "item": item.get("title", "Unnamed item"),
                "error": "No description text found",
                "predicted": None,
                "actual": None,
                "metrics": None
            })
            continue
        
        # Analyze text with the selected model
        predicted = analyze_text(text, vocabularies, api_key, model)
        if not predicted:
            results.append({
                "item": item.get("title", "Unnamed item"),
                "error": "Analysis failed",
                "predicted": None,
                "actual": None,
                "metrics": None
            })
            continue
        
        # Extract actual metadata
        actual = extract_actual_metadata(item, vocabularies)
        
        # Calculate metrics
        metrics = calculate_metrics(predicted, actual)
        
        # Update overall counters
        for prop_metrics in metrics.values():
            if prop_metrics != metrics.get("overall"):
                total_true_positives += prop_metrics.get("true_positives", 0)
                total_false_positives += prop_metrics.get("false_positives", 0)
                total_false_negatives += prop_metrics.get("false_negatives", 0)
        
        # Store results
        results.append({
            "item": item.get("title", "Unnamed item"),
            "text": text,
            "predicted": predicted,
            "actual": actual,
            "metrics": metrics
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate overall metrics across all items
    if total_true_positives + total_false_positives > 0:
        overall_precision = total_true_positives / (total_true_positives + total_false_positives)
    else:
        overall_precision = 0
        
    if total_true_positives + total_false_negatives > 0:
        overall_recall = total_true_positives / (total_true_positives + total_false_negatives)
    else:
        overall_recall = 0
        
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0
    
    # Add overall metrics to results
    results.append({
        "item": "OVERALL METRICS",
        "text": None,
        "predicted": None,
        "actual": None,
        "metrics": {
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "true_positives": total_true_positives,
                "false_positives": total_false_positives,
                "false_negatives": total_false_negatives
            }
        }
    })
    
    return results

# API Key input with default from environment variable
api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")

# Model selection dropdown
model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "ft:gpt-4.1-mini-2025-04-14:personal:klassifizierungstool-test1:Bi0sQKLf"]
selected_model = st.selectbox("Select OpenAI Model", model_options, index=3)  # Default to gpt-4.1-mini (index 3)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Single Text Analysis", "Batch JSON Analysis"])

with tab1:
    # Text input area
    user_text = st.text_area("Enter text to analyze", height=200)
    
    # Vocab URLs input area with default values
    default_vocabs = """https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/index.json
https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/educationalContext/index.json"""
    
    vocab_urls = st.text_area("Vocabulary URLs (one per line)", value=default_vocabs, height=100)
    
    # Start button to trigger analysis
    # In tab1 (Single Text Analysis)
    if st.button("Start Analysis"):
        if not user_text:
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Loading vocabularies..."):
                vocabularies = fetch_vocabularies(vocab_urls)
            
            if vocabularies:
                with st.spinner(f"Analyzing text with {selected_model}..."):
                    result = analyze_text(user_text, vocabularies, api_key, selected_model)
                    
                if result:
                    st.subheader("Analysis Results")
                    
                    for property_analysis in result.get("analysis", []):
                        property_name = property_analysis.get("property", "")
                        st.markdown(f"### {property_name}")
                        
                        matches = property_analysis.get("matches", [])
                        if matches:
                            for match in matches:
                                value = match.get("value", "")
                                id_value = match.get("id", "")
                                confidence = match.get("confidence", 0)
                                explanation = match.get("explanation", "")
                                
                                st.markdown(f"**{value}** (ID: `{id_value}`)")
                                st.progress(min(confidence/100, 1.0))
                                st.markdown(f"Confidence: **{confidence}%**")
                                st.markdown(f"Explanation: {explanation}")
                                st.markdown("---")
                        else:
                            st.markdown("No matches found for this property.")

with tab2:
    # JSON file upload
    uploaded_file = st.file_uploader("Upload JSON file with educational content", type=["json"])
    
    # Vocab URLs input area with default values (same as tab1)
    batch_vocab_urls = st.text_area("Vocabulary URLs (one per line)", value=default_vocabs, height=100, key="batch_vocab_urls")
    
    # Add number input for limiting the number of items to analyze
    max_items = st.number_input("Maximum number of items to analyze (0 = analyze all)", min_value=0, value=0, step=1)
    
    # Start button for batch analysis
    if st.button("Start Batch Analysis"):
        if uploaded_file is None:
            st.warning("Please upload a JSON file")
        else:
            try:
                # Load JSON content
                content_items = json.load(uploaded_file)
                if not isinstance(content_items, list):
                    content_items = [content_items]  # Convert single item to list
                
                total_items = len(content_items)
                st.info(f"Loaded {total_items} content items from JSON file")
                
                # Limit the number of items if specified
                if max_items > 0 and max_items < total_items:
                    content_items = content_items[:max_items]
                    st.info(f"Analyzing first {max_items} of {total_items} items")
                
                # Load vocabularies
                with st.spinner("Loading vocabularies..."):
                    vocabularies = fetch_vocabularies(batch_vocab_urls)
                
                if vocabularies:
                    # In tab2, update the analyze_batch call
                    # Analyze batch
                    with st.spinner(f"Analyzing content items with {selected_model}..."):
                        batch_results = analyze_batch(content_items, vocabularies, api_key, selected_model)
                    
                    # Display summary metrics
                    st.subheader("Summary Metrics")
                    
                    # First, display the overall metrics item if it exists
                    overall_result = next((r for r in batch_results if r["item"] == "OVERALL METRICS"), None)
                    if overall_result and overall_result["metrics"]:
                        # Display overall metrics across all properties and items
                        st.markdown("### Overall Metrics (All Properties, All Items)")
                        metrics = overall_result["metrics"]["overall"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Precision", f"{metrics['precision']:.2%}")
                        with col2:
                            st.metric("Recall", f"{metrics['recall']:.2%}")
                        with col3:
                            st.metric("F1 Score", f"{metrics['f1']:.2%}")
                        
                        # Display counts
                        st.markdown("#### Counts Across All Items")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("True Positives", metrics["true_positives"])
                        with col2:
                            st.metric("False Positives", metrics["false_positives"])
                        with col3:
                            st.metric("False Negatives", metrics["false_negatives"])
                        
                        # Calculate and display metrics per property across all items
                        st.markdown("### Metrics Per Property (Across All Items)")
                        
                        # Collect all property names from all results
                        all_properties = set()
                        for result in batch_results:
                            if result["metrics"]:
                                for prop_name in result["metrics"]:
                                    if prop_name != "overall":
                                        all_properties.add(prop_name)
                        
                        # Calculate metrics for each property across all items
                        property_metrics = {}
                        for prop_name in all_properties:
                            true_positives = 0
                            false_positives = 0
                            false_negatives = 0
                            
                            for result in batch_results:
                                if result["metrics"] and prop_name in result["metrics"]:
                                    prop_metrics = result["metrics"][prop_name]
                                    true_positives += prop_metrics.get("true_positives", 0)
                                    false_positives += prop_metrics.get("false_positives", 0)
                                    false_negatives += prop_metrics.get("false_negatives", 0)
                            
                            # Calculate precision, recall, and F1 for this property
                            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            
                            property_metrics[prop_name] = {
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "true_positives": true_positives,
                                "false_positives": false_positives,
                                "false_negatives": false_negatives
                            }
                        
                        # Display metrics for each property
                        for prop_name, metrics in property_metrics.items():
                            with st.expander(f"Property: {prop_name}", expanded=True):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Precision", f"{metrics['precision']:.2%}")
                                with col2:
                                    st.metric("Recall", f"{metrics['recall']:.2%}")
                                with col3:
                                    st.metric("F1 Score", f"{metrics['f1']:.2%}")
                                
                                # Display counts
                                st.markdown("#### Counts")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("True Positives", metrics["true_positives"])
                                with col2:
                                    st.metric("False Positives", metrics["false_positives"])
                                with col3:
                                    st.metric("False Negatives", metrics["false_negatives"])
                    
                    # Then display individual items
                    st.subheader("Detailed Results Per Item")
                    
                    for i, result in enumerate(batch_results):
                        if result["item"] == "OVERALL METRICS":
                            continue  # Skip, already displayed above
                            
                        with st.expander(f"Item {i+1}: {result['item']}"):
                            if "error" in result and result["error"]:
                                st.error(result["error"])
                                continue
                            
                            st.markdown("### Text Analyzed")
                            st.text(result["text"])
                            
                            # Display comparison table
                            st.markdown("### Metadata Comparison")
                            
                            if result["predicted"] and result["actual"]:
                                # Get all unique property names
                                all_properties = set()
                                for prop_analysis in result["predicted"].get("analysis", []):
                                    all_properties.add(prop_analysis.get("property", ""))
                                for prop_name in result["actual"].keys():
                                    all_properties.add(prop_name)
                                
                                # Create comparison for each property
                                for prop_name in all_properties:
                                    st.markdown(f"#### {prop_name}")
                                    
                                    # Get predicted values
                                    predicted_analysis = next((p for p in result["predicted"].get("analysis", []) 
                                                            if p["property"] == prop_name), None)
                                    predicted_matches = predicted_analysis.get("matches", []) if predicted_analysis else []
                                    
                                    # Get actual values
                                    actual_matches = result["actual"].get(prop_name, [])
                                    
                                    # Extract IDs for comparison
                                    predicted_ids = set(match["id"] for match in predicted_matches)
                                    actual_ids = set(match["id"] for match in actual_matches)
                                    
                                    # True positives (correctly predicted)
                                    true_positive_ids = predicted_ids.intersection(actual_ids)
                                    # False positives (incorrectly predicted)
                                    false_positive_ids = predicted_ids - actual_ids
                                    # False negatives (missed actual values)
                                    false_negative_ids = actual_ids - predicted_ids
                                    
                                    # Display true positives
                                    if true_positive_ids:
                                        st.markdown("**✓ Correctly Predicted:**")
                                        for match in predicted_matches:
                                            if match["id"] in true_positive_ids:
                                                st.markdown(f"- {match['value']} (ID: `{match['id']}`)")
                                    
                                    # Display false positives
                                    if false_positive_ids:
                                        st.markdown("**❌ Incorrectly Predicted:**")
                                        for match in predicted_matches:
                                            if match["id"] in false_positive_ids:
                                                st.markdown(f"- {match['value']} (ID: `{match['id']}`)")
                                    
                                    # Display false negatives
                                    if false_negative_ids:
                                        st.markdown("**❓ Missed Actual Values:**")
                                        for match in actual_matches:
                                            if match["id"] in false_negative_ids:
                                                st.markdown(f"- {match['value']} (ID: `{match['id']}`)")
                                    
                                    # Display counts
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("True Positives", len(true_positive_ids))
                                    with col2:
                                        st.metric("False Positives", len(false_positive_ids))
                                    with col3:
                                        st.metric("False Negatives", len(false_negative_ids))
                            else:
                                if not result["predicted"]:
                                    st.warning("No predicted metadata available")
                                if not result["actual"]:
                                    st.warning("No actual metadata found in the content item")
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")