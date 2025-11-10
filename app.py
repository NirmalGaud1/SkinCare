# app.py
import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from collections import defaultdict
import random
import time
import os

# ========================================
# CONFIG (2025 UPDATE)
# ========================================
st.set_page_config(page_title="Skincare Trend Detector", layout="wide")

# NOTE: Replace with secure environment variable usage in production
GEMINI_API_KEY = "AIzaSyANsbK02NgbT67D16WEmHwq1-f3jZLH4PU"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"
DATASET_PATH = "./dummy_skincare_text.csv"
OUTPUT_JSON = "./structured_trends_gemini.json"
BATCH_SIZE = 20

# ========================================
# HELPER: GENERATE DUMMY DATASET
# ========================================
@st.cache_data
def generate_dataset():
    random.seed(42)
    trend_seeds = [
        ("ceramides", "ingredient-driven", "ceramides", "serum", "skin barrier"),
        ("slugging", "technique/routine", None, "occlusive", "dryness"),
        ("microbiome-friendly", "ingredient-driven", "probiotics", "cleanser", "acne"),
        ("anti-pollution skincare", "problem-solving innovation", "antioxidants", "mist", "pollution"),
        ("skinimalism", "cultural shift", None, None, "routine overload"),
        ("polyglutamic acid", "ingredient-driven", "polyglutamic acid", "serum", "hydration"),
        ("bakuchiol", "ingredient-driven", "bakuchiol", "serum", "anti-aging"),
        ("niacinamide", "ingredient-driven", "niacinamide", "serum", "oiliness"),
        ("skin flooding", "technique/routine", None, "multi-serum", "dehydration"),
        ("glass skin", "cultural shift", None, "multi-step", "dullness"),
    ]
    templates = [
        "Just tried {trend} and my skin feels amazing!",
        "Is {trend} worth the hype? Asking for a friend.",
        "Loving how {trend} fixed my {concern} issue.",
        "Everyone on TikTok is doing {trend} now.",
        "Switched to {trend} and never going back.",
        "Heard {ingredient} in {product} is the new holy grail.",
        "Finally found a {product} with {ingredient} that works!",
        "My dermatologist recommended {trend} for {concern}.",
        "This {trend} routine changed my life.",
        "Why is no one talking about {trend}?",
    ]
    noise_texts = [
        "Best sunscreen ever!", "This moisturizer smells great.",
        "Cleansing balm removed all my makeup.", "Retinol burns my skin.",
        "Need a good eye cream.", "My skin is glowing today!",
        "Hydrating mask on point.", "Love the packaging of this serum.",
        "Finally found a non-comedogenic foundation.", "SPF 50 is a must in summer.",
    ]
    emoji = ["", " [sparkles]", " [fire]", " [100]", " [heart eyes]"]

    data = []
    for i in range(1, 201):
        if random.random() < 0.75:
            trend, _, ing, prod, conc = random.choice(trend_seeds)
            text = random.choice(templates).format(
                trend=trend, ingredient=ing or "", product=prod or "product", concern=conc or "skin"
            )
        else:
            text = random.choice(noise_texts)
        if random.random() < 0.3:
            text += random.choice(emoji)
        data.append({"id": i, "text": text.strip()})

    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    return df

# ========================================
# UI
# ========================================
st.title("🧴 Skincare Trend Detection POC")
st.markdown("**AI-Powered Trend Extraction using Google Gemini 2.0**")

# Load or generate dataset
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
    st.success(f"Loaded dataset: `{DATASET_PATH}` → {len(df)} rows")
else:
    st.info("Generating dummy dataset...")
    df = generate_dataset()
    st.success(f"Generated `{DATASET_PATH}` → {len(df)} rows")

# Preview
with st.expander("📊 View Raw Data (First 10 rows)", expanded=False):
    st.dataframe(df.head(10), width='stretch')

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("🚀 Run Trend Detection", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    status_text.text("Processing batches...")
    all_results = []
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    # GEMINI PROMPT (refined for 2.0; escaped braces)
    GEMINI_PROMPT = """
You are a senior skincare trend analyst. Analyze the text snippets and:

1. Identify emerging trends (repeated/excited phrases).
2. For each trend, extract:
    - ingredient (null if none)
    - benefit
    - product_type (null if none)
    - target_concern
3. Classify into one:
    - "ingredient-driven"
    - "technique/routine"
    - "problem-solving innovation"
    - "cultural shift"
    - "fad"

Return ONLY a JSON array with exact schema (no extra text):
[
  {{
    "trend": "string",
    "attributes": {{
      "ingredient": "string or null",
      "benefit": "string",
      "product_type": "string or null",
      "target_concern": "string"
    }},
    "category": "string",
    "evidence_count": 1,
    "sample_texts": ["quote1", "quote2"]
  }}
]

Snippets:
{snippets}
"""

    def extract_batch(batch_df, batch_idx):
        snippets = "\n".join([f"- ID{r['id']}: {r['text']}" for _, r in batch_df.iterrows()])
        prompt = GEMINI_PROMPT.format(snippets=snippets)
        
        # --- SAFETY CONFIGURATION TO PREVENT BLOCKING BENIGN HEALTH/SKIN TOPICS ---
        safety_config = [
            # FIX for AttributeError: Use genai.SafetySetting directly
            genai.SafetySetting( 
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            )
        ]
        
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=2048,
                    safety_settings=safety_config
                )
            )

            # --- GUARDRAIL FIX FOR ATTRIBUTE ERROR (response.text check) ---
            if not hasattr(response, 'text') or not response.text:
                # Detailed failure report
                finish_reason = response.candidates[0].finish_reason.name if (response.candidates and response.candidates[0]) else 'NO_CANDIDATE'
                prompt_blocked = response.prompt_feedback.block_reason.name if response.prompt_feedback and response.prompt_feedback.block_reason.name != 'SAFETY_REASON_UNSPECIFIED' else 'NO_BLOCK'
                
                st.warning(f"Batch {batch_idx}: No valid text returned. Finish Reason: **{finish_reason}**. Prompt Blocked: {prompt_blocked}")
                return []
            # ----------------------------------------------------------------

            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw.split("```json", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            parsed = json.loads(raw.strip())
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            st.error(f"Gemini error in batch {batch_idx}: {e}")
            return []

    # Process
    for i in range(0, len(df), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        batch = df.iloc[i:i+BATCH_SIZE]
        status_text.text(f"Processing batch {batch_idx}/{total_batches}...")
        progress_bar.progress((i + len(batch)) / len(df))
        batch_results = extract_batch(batch, batch_idx)
        all_results.extend(batch_results)
        time.sleep(1.5)

    # Aggregate
    status_text.text("Aggregating results...")
    agg = defaultdict(lambda: {"evidence_count": 0, "sample_texts": []})
    for item in all_results:
        if isinstance(item, dict):
            trend = item.get("trend", "").strip()
            cat = item.get("category", "")
            key = (trend.lower(), cat)
            entry = agg[key]
            entry["evidence_count"] += item.get("evidence_count", 1)
            entry["sample_texts"].extend(item.get("sample_texts", []))
            entry["sample_texts"] = entry["sample_texts"][:3]
            if "trend" not in entry:
                entry.update({"trend": trend, "attributes": item.get("attributes", {}), "category": cat})

    final_trends = sorted([
        {"trend": v["trend"], "attributes": v["attributes"], "category": v["category"],
         "evidence_count": v["evidence_count"], "sample_texts": v["sample_texts"]}
        for v in agg.values() if v["trend"]
    ], key=lambda x: x["evidence_count"], reverse=True)

    # Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_trends, f, indent=2, ensure_ascii=False)

    progress_bar.empty()
    status_text.empty()

    # Results
    with results_container:
        st.success(f"POC Complete! → `{OUTPUT_JSON}` ({len(final_trends)} trends detected)")
        if final_trends:
            st.subheader("🔍 Top Detected Trends")

            cols = st.columns(3)
            for idx, trend in enumerate(final_trends[:6]):
                with cols[idx % 3]:
                    st.markdown(f"### {idx+1}. **{trend['trend']}**")
                    st.caption(f"**Category:** {trend['category']}")
                    st.metric("Evidence Count", trend['evidence_count'])
                    with st.expander("Details"):
                        st.json(trend["attributes"])
                        st.write("**Sample Texts:**")
                        for txt in trend["sample_texts"]:
                            st.caption(txt)

            # Full table
            with st.expander("📋 All Trends (Table)"):
                display_df = pd.DataFrame(final_trends)
                st.dataframe(display_df, width='stretch')

            # Download
            st.download_button(
                label="💾 Download JSON Results",
                data=json.dumps(final_trends, indent=2),
                file_name="structured_trends_gemini.json",
                mime="application/json"
            )
        else:
            st.warning("No trends detected – check API key or try a smaller batch.")

else:
    st.info("👆 Click **Run Trend Detection** in the sidebar to start.")
    st.markdown("---")
    st.markdown("""
### 📝 How It Works
1. **Dataset**: 200 synthetic skincare posts/reviews (generated if missing).  
2. **Gemini 2.0 Analysis**: Batches texts → Extracts trends, attributes, categories.  
3. **Output**: Structured JSON with prioritized trends (e.g., ceramides as "ingredient-driven").  
4. **POC Fit**: Demonstrates trend detection for skincare brand MVP.
    """)

# Footer
st.markdown("---")
st.caption("*Updated for Gemini 2.0 (Nov 2025). Revoke API key after use.*")
