# app_gpt2.py
import streamlit as st
import pandas as pd
import json
import random
import os
from collections import defaultdict, Counter
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Skincare Trend Detector (GPT-2)", layout="wide")

DATASET_PATH = "./dummy_skincare_text.csv"
OUTPUT_JSON   = "./structured_trends_gpt2.json"
BATCH_SIZE    = 20

# -------------------------------------------------
# 1. GENERATE DUMMY DATASET (same as before)
# -------------------------------------------------
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
    noise = [
        "Best sunscreen ever!", "This moisturizer smells great.",
        "Cleansing balm removed all my makeup.", "Retinol burns my skin.",
        "Need a good eye cream.", "My skin is glowing today!",
    ]
    emojis = ["", " sparkles", " fire", " 100", " heart eyes"]

    rows = []
    for i in range(1, 201):
        if random.random() < 0.75:
            trend, _, ing, prod, conc = random.choice(trend_seeds)
            txt = random.choice(templates).format(
                trend=trend, ingredient=ing or "", product=prod or "product", concern=conc or "skin"
            )
        else:
            txt = random.choice(noise)
        if random.random() < 0.3:
            txt += random.choice(emojis)
        rows.append({"id": i, "text": txt.strip()})
    df = pd.DataFrame(rows)
    df.to_csv(DATASET_PATH, index=False)
    return df

# -------------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------------
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
    st.success(f"Loaded `{DATASET_PATH}` → {len(df)} rows")
else:
    st.info("Generating dummy dataset...")
    df = generate_dataset()
    st.success(f"Generated `{DATASET_PATH}` → {len(df)} rows")

with st.expander("View Raw Data (first 10)", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------------
# 3. GPT-2 PIPELINE (lightweight)
# -------------------------------------------------
@st.cache_resource
def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model     = AutoModelForCausalLM.from_pretrained("gpt2")
    # Use a text-generation pipeline with forced JSON schema
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=400,
        temperature=0.0,
        return_full_text=False,
    )
    return pipe

pipe = load_gpt2()

# -------------------------------------------------
# 4. PROMPT (same schema, but GPT-2 needs strict instruction)
# -------------------------------------------------
PROMPT_TEMPLATE = """
You are a skincare trend analyst. From the snippets below, return ONLY a JSON array (no markdown, no extra text) with this exact schema:

[
  {
    "trend": "string",
    "attributes": {
      "ingredient": "string or null",
      "benefit": "string",
      "product_type": "string or null",
      "target_concern": "string"
    },
    "category": "one of: ingredient-driven, technique/routine, problem-solving innovation, cultural shift, fad",
    "evidence_count": 1,
    "sample_texts": ["quote1", "quote2"]
  }
]

Snippets:
{snippets}

JSON:
"""

# -------------------------------------------------
# 5. RUN BUTTON
# -------------------------------------------------
if st.sidebar.button("Run Trend Detection (GPT-2)", type="primary"):
    progress = st.progress(0)
    status   = st.empty()
    container = st.container()

    status.text("Processing batches with GPT-2...")
    all_raw = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        snippets = "\n".join([f"- ID{r['id']}: {r['text']}" for _, r in batch.iterrows()])
        prompt = PROMPT_TEMPLATE.format(snippets=snippets)

        # GPT-2 generation
        try:
            out = pipe(prompt)[0]["generated_text"].strip()
            # Clean possible code fences
            if out.startswith("```"):
                out = out.split("```", 2)[-1] if "```" in out else out
            if out.endswith("```"):
                out = out.rsplit("```", 1)[0]
            batch_json = json.loads(out)
            all_raw.extend(batch_json if isinstance(batch_json, list) else [])
        except Exception as e:
            st.warning(f"Batch {i//BATCH_SIZE+1} parsing error: {e}")
        progress.progress((i + len(batch)) / len(df))

    # -------------------------------------------------
    # 6. AGGREGATE
    # -------------------------------------------------
    agg = defaultdict(lambda: {"evidence_count": 0, "sample_texts": []})
    for item in all_raw:
        trend = item.get("trend", "").strip()
        cat   = item.get("category", "")
        key   = (trend.lower(), cat)

        entry = agg[key]
        entry["evidence_count"] += item.get("evidence_count", 1)
        entry["sample_texts"].extend(item.get("sample_texts", []))
        entry["sample_texts"] = entry["sample_texts"][:3]

        if "trend" not in entry:
            entry.update({
                "trend": trend,
                "attributes": item.get("attributes", {}),
                "category": cat
            })

    final = sorted(
        [{"trend": v["trend"], "attributes": v["attributes"], "category": v["category"],
          "evidence_count": v["evidence_count"], "sample_texts": v["sample_texts"]}
         for v in agg.values() if v["trend"]],
        key=lambda x: x["evidence_count"], reverse=True
    )

    # Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    progress.empty()
    status.empty()

    # -------------------------------------------------
    # 7. DISPLAY RESULTS
    # -------------------------------------------------
    with container:
        st.success(f"Done! → `{OUTPUT_JSON}` ({len(final)} trends)")
        if final:
            st.subheader("Top Trends")
            cols = st.columns(3)
            for idx, t in enumerate(final[:6]):
                with cols[idx % 3]:
                    st.markdown(f"### {idx+1}. **{t['trend']}**")
                    st.caption(f"*{t['category']}*")
                    st.metric("Evidence", t['evidence_count'])
                    with st.expander("Details"):
                        st.json(t["attributes"])
                        st.write("**Samples:**")
                        for s in t["sample_texts"]:
                            st.caption(s)

            with st.expander("All Trends (Table)"):
                st.dataframe(pd.DataFrame(final), use_container_width=True)

            st.download_button(
                "Download JSON",
                data=json.dumps(final, indent=2),
                file_name="structured_trends_gpt2.json",
                mime="application/json"
            )
        else:
            st.warning("No trends extracted – try a smaller batch or check prompt.")

else:
    st.info("Click **Run Trend Detection (GPT-2)** in the sidebar.")
    st.markdown("---")
    st.markdown("""
### How It Works
1. **Dataset** – 200 synthetic skincare posts (generated automatically).  
2. **GPT-2** – Local open-source LLM extracts trends, attributes, category.  
3. **Output** – `structured_trends_gpt2.json` (same format as Gemini version).  
4. **No API key** – 100% offline after model download.
    """)
