import os
import json
from openai import OpenAI
import time

# --- CONFIGURATION ---

# 1. Set up the OpenAI client. It will use the OPENAI_API_KEY environment variable.
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key= api_key
)

# 2. Populate this list with all the religions/philosophies you want to score.
MASTER_RELIGION_LIST = []
with open("all_rels_da.txt", "r", encoding="utf-8") as f:
    MASTER_RELIGION_LIST = f.readlines()
MASTER_RELIGION_LIST = [line.strip() for line in MASTER_RELIGION_LIST]
# 3. Define the output .jsonl file.
OUTPUT_FILENAME = "religion_scores.jsonl"


# --- PROMPT DEFINITION ---

def generate_system_prompt():
    """Defines the core instruction for the language model to force JSON output."""
    return """
You are a data entry assistant specializing in religious studies. Your task is to analyze a given religion based on a specific 10-dimension rubric and output ONLY a valid JSON object containing the scores.

Here is the scoring rubric:
* Metaphysics: 1 (Purely Materialist) → 10 (Strongly Transcendent/Supernatural)
* Cosmology: 1 (Purely Naturalistic) → 10 (Direct Supernatural Creation)
* Anthropology: 1 (Purely Biological) → 10 (Possessing a Divine/Eternal Soul)
* Soteriology: 1 (This-Worldly/Self-Achieved) → 10 (Requires Divine Grace/Intervention)
* Ethics: 1 (Human-Derived/Situational) → 10 (Divinely Commanded/Absolute)
* Praxis: 1 (Minimal/No Ritual) → 10 (Highly Prescribed Rituals)
* Epistemology: 1 (Purely Empirical/Rational) → 10 (Primarily based on Divine Revelation)
* Ecclesiology: 1 (Decentralized / Individualistic) → 10 (Rigid Hierarchy)
* Eschatology: 1 (Cyclical / A-historical) → 10 (Aggressively Universalist)
* Exclusivity: 1 (Pluralist) → 5 (Inclusivist) → 10 (Exclusivist)

Based on your training data, provide an integer score from 1 to 10 for each dimension for the religion provided by the user. The output must be a single, valid JSON object with keys matching the dimension names and integer values for the scores. Do not include any other text, explanation, or markdown.
Ensure that the assigned scores are based on the provided rubric and reflect the characteristics of the religion accurately as possible.
"""


# --- MAIN GENERATION LOOP ---

def generate_scores():
    """
    Generates a score object for each religion and appends it to a .jsonl file.
    """
    system_prompt = generate_system_prompt()

    # Open the output file once in append mode.
    with open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
        for religion in MASTER_RELIGION_LIST:
            print(f"Processing: {religion}")
            try:
                # 1. Call the OpenAI API using JSON Mode
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"}, # Enable JSON Mode
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": religion}
                    ]
                )
                
                # The response content will be a valid JSON string
                scores_json_string = completion.choices[0].message.content
                scores_dict = json.loads(scores_json_string)

                # 2. Structure the data for the .jsonl file
                output_entry = {
                    "religion": religion,
                    "scores": scores_dict
                }

                # 3. Convert to a JSON string and append it as a new line
                json_line = json.dumps(output_entry, ensure_ascii=False)
                f.write(json_line + '\n')
                
                print(f"  -> Successfully saved scores for {religion}")

            except Exception as e:
                print(f"  -> An error occurred while processing {religion}: {e}")
                # Optional: Add a small delay before the next attempt
                time.sleep(2)

# --- EXECUTE SCRIPT ---
if __name__ == "__main__":
    generate_scores()
    print(f"\nProcessing complete. Data saved to {OUTPUT_FILENAME}")