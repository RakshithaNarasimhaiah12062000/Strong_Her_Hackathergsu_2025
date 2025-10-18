# phase3_app.py

import streamlit as st
from phase2 import create_rag_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import json
# --- Add these imports ---
import pandas as pd
import numpy as np
import random


# --- Page config ---
st.set_page_config(page_title="StrongHer", page_icon="🌼", layout="wide")


# ---- Global style: remove mysterious top bar + tidy spacing ----
# ---- Global style (gradient, components, and hide Streamlit chrome) ----
st.markdown("""
<style>

/* Layout padding */
.block-container { padding-top: 1rem !important; }

/* Kill separators that can look like bars */
hr, [data-testid="stDivider"], div[role="separator"] { display: none !important; }

/* Remove Streamlit chrome (top bar, toolbar, bottom status/footer, and the rounded "decoration" pills) */
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
footer,
div[data-testid="stStatusWidget"],
div[data-testid="stDecoration"] {    /* <- this is the key one */
  display: none !important;
}

/* Hide any empty text inputs (defensive) */
div.stTextInput label:empty,
div.stTextInput input[aria-label=""] { display: none !important; }

/* App background */
.stApp {
  background: linear-gradient(135deg, #9d4edd 0%, #c48aff 40%, #f7a8f8 75%, #ffb6c1 100%);
  background-attachment: fixed;
  color: #fff !important;
}

/* Home: quote card */
.quote-wrap {
  display: inline-block;
  padding: 22px 24px;
  border-radius: 22px;
  background: rgba(255,255,255,0.22);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 14px 50px rgba(0,0,0,0.25);
}
.hero h1 { font-size: 3.3rem; line-height: 1.08; margin: 0 0 .35rem 0; color: #fff; }
.hero .sub { font-size: 1.15rem; opacity: .95; margin-bottom: 1.1rem; color: #fff; }
.quote-text {
  font-size: 2.4rem; font-weight: 900; line-height: 1.28;
  background: linear-gradient(90deg, #ffffff 0%, #fff7b0 40%, #ffe0f7 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.mini { margin-top: 12px; color: #fff; opacity: .95; }

/* Buttons */
.stButton button{
  background-color: rgba(255,255,255,0.18);
  color: #fff; border: 1px solid rgba(255,255,255,0.35);
  border-radius: 12px; padding: 10px 18px; font-weight: 700;
}
.stButton button:hover{ background-color: rgba(255,255,255,0.32); color: #1a1a1a; }

/* Footer */
.footer{ text-align:center; font-size:13px; color:#fff; opacity:.9; padding:10px 0; }

/* Parenting Hub dark panels */
.dark-card {
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 20px;
  padding: 22px 22px;
  color: #ffffff !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.dark-card h1, .dark-card h2, .dark-card h3, .dark-card p, .dark-card label { color: #ffffff !important; }
.dark-card .stTextInput input, .dark-card .stTextArea textarea, .dark-card .stNumberInput input {
  background: rgba(255,255,255,0.08) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 12px !important;
}
.dark-card .stSelectbox div[data-baseweb="select"] > div,
.dark-card .stRadio div[role="radiogroup"] {
  background: rgba(255,255,255,0.08) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 12px !important;
}

/* Tabs (remove red accent, use lilac) */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid rgba(255,255,255,0.15) !important;
}
.stTabs [data-baseweb="tab"]::before,
.stTabs [data-baseweb="tab"]::after { background: none !important; border: none !important; box-shadow: none !important; }
.stTabs [data-baseweb="tab"] {
  position: relative; color: #f6f3ff !important; border-radius: 12px; padding: 10px 14px; background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: rgba(0,0,0,0.35); color: #fff !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {
  content: ""; position: absolute; left: 12px; right: 12px; bottom: -1px; height: 3px; border-radius: 3px;
  background: #d4b3ff !important;
}

/* Small spacing */
.ph-section { margin-bottom: 18px; }

/* --- Smart Meal Helper cards --- */
.meal-card {
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 18px;
  padding: 16px 18px;
  margin: 10px 0;
  color: #fff;
  box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}
.meal-card h4 {
  margin: 0 0 6px 0;
  font-size: 1.1rem;
  letter-spacing: .2px;
}
.meal-meta { opacity:.9; font-size:.92rem; margin-bottom:8px; }
.meal-badges span{
  display:inline-block; padding:6px 10px; border-radius:999px;
  background: rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18);
  margin-right:6px; margin-bottom:6px; font-size:.82rem;
}
.meal-notes { opacity:.95; }
.meal-actions { margin-top:8px; }
.energy-tip {
  display:inline-block; padding:10px 14px; border-radius:12px;
  background: rgba(0,0,0,.45); border:1px solid rgba(255,255,255,.18);
  margin: 10px 0; font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# Load RAG / LLM
@st.cache_resource
def load_chain():
    return create_rag_chain()

qa_chain = load_chain()
search = DuckDuckGoSearchRun()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# ------------------ Helper to call RAG ------------------
def ask_chat(question: str) -> str:
    """Safely query the RAG chain and return an answer."""
    try:
        if not question.strip():
            return "Please enter a valid question 💬"
        return qa_chain.run(question)
    except Exception as e:
        return f"⚠️ Sorry, something went wrong: {e}"

# ---------- Meals & Boosters data ----------
@st.cache_data
def load_meal_data():
    meals_path = "static_content/meals.csv"
    boosters_path = "static_content/boosters.csv"  # optional

    try:
        meals_df = pd.read_csv(meals_path)
    except FileNotFoundError:
        meals_df = pd.DataFrame()

    try:
        boosters_df = pd.read_csv(boosters_path)
    except FileNotFoundError:
        boosters_df = pd.DataFrame()

    # normalize a few columns for safety
    for col in ["diet", "meal_type", "tags", "title", "notes", "recipe"]:
        if col in meals_df.columns:
            meals_df[col] = meals_df[col].fillna("").astype(str)

    # Add protein density for sorting
    if not meals_df.empty:
        meals_df["protein_density"] = (
            meals_df["protein_g"].replace(0, np.nan) / meals_df["cals"].replace(0, np.nan)
        ).fillna(0)

    return meals_df, boosters_df

MEALS_DF, BOOSTERS_DF = load_meal_data()

# Load static content JSONs
@st.cache_data
def load_static_content():
    base = "static_content"
    guides = {}
    nutrition = {}
    templates = {}
    how_to = {}
    try:
        with open(f"{base}/guides.json", "r", encoding="utf-8") as f:
            guides = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(f"{base}/nutrition.json", "r", encoding="utf-8") as f:
            nutrition = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(f"{base}/templates.json", "r", encoding="utf-8") as f:
            templates = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(f"{base}/how_to.json", "r", encoding="utf-8") as f:
            how_to = json.load(f)
    except FileNotFoundError:
        pass
    return guides, nutrition, templates, how_to

guides_data, nutrition_data, templates_data, how_to_data = load_static_content()

# Navigation / Routing (FULL)
# -----------------------------
PAGES = {
    "🏠 Daily Boost": "home",                   # Emotional & mental health anchor — sets the tone
    "🤱 Motherhood Corner": "parenting",        # Core feature — guidance, wellness, and maternal care
    "🌼 Postpartum Wellness": "postpartum",     # Direct link to physical & emotional recovery
    "🥗 Smart Meal Planner": "meals",           # Practical, health-focused, and interactive
    "📍 Community": "community",      # Social well-being & support — critical for single moms
    "⚖️ Safety & Rights": "legal",              # Empowerment & protection resources
    "💵 Financial Wellness": "finance",         # Stability & long-term well-being (still important but secondary)
}

# Initialize current page once
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# --- Navigation helpers ---
# Reverse map: code -> label
PAGE_LABEL_BY_CODE = {v: k for k, v in PAGES.items()}

# Apply any pending navigation BEFORE rendering the sidebar radio
if "pending_nav" in st.session_state:
    st.session_state["page"] = st.session_state.pop("pending_nav")

# --- Sidebar / menu (index-driven, no key mutation) ---
labels = list(PAGES.keys())
current_label = PAGE_LABEL_BY_CODE[st.session_state["page"]]

# --- SIDEBAR HEADER / BRANDING ---
st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f7f3ff;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sidebar-sub {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        text-align: center;
        margin-bottom: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Optional logo (small icon)
st.sidebar.markdown(
    """
    <div class="sidebar-title">🌼 StrongHer 🌼</div>
    <div class="sidebar-sub">Empower. Heal. Rise.</div>
    """,
    unsafe_allow_html=True,
)

choice = st.sidebar.radio(
    "Navigate to:",
    labels,
    index=labels.index(current_label),
)
st.session_state["page"] = PAGES[choice]

# Current page code for routing below
page = st.session_state["page"]

# --- Page: Home / Daily Boost ---
if page == "home":
    from PIL import Image
    import random
    img = Image.open("assets/img.png")

    # Two-column layout (text on left, image on right)
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        # Hero Section Text
        st.markdown(
            """
            <div style="text-align:left; margin-bottom:2rem;">
                <h1 style="font-size:3rem; color:white; margin-bottom:0.3rem;">
                    Welcome to <b>StrongHer</b>
                </h1>
                <p style="font-size:1.2rem; color:white; opacity:0.9;">
                    Everyday support for single mothers 💜
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- QUOTE RANDOMIZER ---
        quotes = [
            ("“You are stronger than you think.”", "Pause and take 4 slow breaths.", "Micro-goal: drink one full glass of water. 💧"),
            ("“One small step today is still progress.”", "Focus on just one thing you can do right now.", "Micro-goal: stand, stretch, and smile. 🌼"),
            ("“Rest is productive.”", "You don’t need to earn rest — you deserve it.", "Micro-goal: take 5 minutes just to breathe. 🌙"),
            ("“You are doing enough.”", "Let go of guilt — your effort counts.", "Micro-goal: write down one win today. 📝"),
            ("“Grace over perfection.”", "Show yourself the same kindness you give others.", "Micro-goal: drink some water and close your eyes for 30 seconds. 💧"),
            ("“Breathe. You’ve got this.”", "No one can do it all, all the time.", "Micro-goal: take a deep breath and unclench your jaw. 🌬️"),
            ("“It’s okay to start over.”", "Every sunrise is a reset button.", "Micro-goal: step outside and feel the air. ☀️"),
            ("“You’re not alone.”", "Even on tough days, you are part of something bigger.", "Micro-goal: send a quick text to someone you love. 💌"),
            ("“Celebrate the small victories.”", "Each little win adds up.", "Micro-goal: write one thing that made you smile today. 🌸"),
            ("“Your pace is perfect.”", "Healing and growth aren’t races.", "Micro-goal: take a slow sip of tea or coffee. ☕"),
        ]

        # Initialize quote index
        if "quote_idx" not in st.session_state:
            st.session_state.quote_idx = random.randint(0, len(quotes) - 1)

        quote, tip, micro_goal = quotes[st.session_state.quote_idx]

        # Display quote card
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.18);
            border-radius: 20px;
            padding: 1.8rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
            color: white;
            backdrop-filter: blur(10px);
            max-width: 480px;
            ">
            <p style="font-size:1.5rem; font-weight:600; margin-bottom:1rem;">
                🌸 {quote}
            </p>
            <ul style="list-style-type:none; padding-left:0; font-size:1.1rem; line-height:1.6;">
                <li>{tip}</li>
                <li>{micro_goal}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # --- Small regenerate button below quote ---
        st.markdown("""
        <style>
        .regen-btn {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 30px;
            padding: 6px 14px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 12px;
            transition: all 0.2s ease-in-out;
        }
        .regen-btn:hover {
            background: rgba(255,255,255,0.35);
            color: #1a1a1a;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create a smaller centered button
        col1, col2, col3 = st.columns([1, 0.8, 1])
        with col2:
            if st.button("Regenerate Boost", key="regen_quote_btn"):
                st.session_state.quote_idx = random.randint(0, len(quotes) - 1)
                st.rerun()

    with right_col:
        st.image(
            img,
            use_column_width=True
        )

    # Spacer for layout adjustment
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # CTAs → set pending_nav BEFORE sidebar renders next run
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Chat"):
            st.session_state["pending_nav"] = "parenting"   # go to Parenting Hub
            st.rerun()

    with c2:
        if st.button("Find Local Help"):
            st.session_state["pending_nav"] = "community"   # go to Community Finder
            st.rerun()

# --- Page: Parenting Hub ---
elif page == "parenting":
    # ----- OPTIONAL: tiny helper CSS for the tip cards (keeps style self-contained) -----
    st.markdown("""
    <style>
      .tip-card {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 14px;
        padding: 10px 14px;
        margin: 6px 0;
      }
      .tip-header { font-weight: 700; margin-bottom: 6px; }
      .muted { opacity: .85; }
    </style>
    """, unsafe_allow_html=True)

    # -------------------- LAYOUT --------------------
    left, right = st.columns([2, 1], gap="large")

    # -------------------- LEFT: Q&A --------------------
    with left:
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.markdown('<h1 style="margin:0 0 8px 0;">🤱 Motherhood Corner</h1>', unsafe_allow_html=True)
        st.markdown('<h3 class="ph-section" style="margin:0 0 10px 0;">Ask your parenting questions</h3>', unsafe_allow_html=True)

        question = st.text_input("Type your question:", key="parenting_q")
        if question:
            answer = ask_chat(question)
            st.markdown('<div class="ph-section" style="height:8px;"></div>', unsafe_allow_html=True)
            st.markdown("**Answer:**")
            st.write(answer)
        else:
            st.markdown('<div class="muted">Tip: Ask about sleep routines, tantrums, potty training, or milestones.</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- RIGHT: Guides / Nutrition / How-to --------------------
    with right:
        st.markdown('<div class="dark-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="margin:0 0 10px 0;">Helpful Libraries</h2>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Quick Guides", "Nutrition", "Show Me How"])

        # ---------- TAB 1: QUICK GUIDES ----------
        with tab1:
            guides_catalog = {
                "Infant (0–6 mo)": [
                    "🍼 Feed on demand — watch hunger cues (rooting, hand-to-mouth).",
                    "🧸 Daily tummy time (start with 2–3 min, build up).",
                    "😴 Safe sleep: on the **back**, firm mattress, no loose items.",
                    "👩‍🍼 Bond: soft talking, humming, gentle touch.",
                    "🩺 Track: wet diapers, growth, vaccines."
                ],
                "Toddler (1–3 yr)": [
                    "🥣 Small frequent meals; offer choices (2 healthy options).",
                    "🚫 Baby-proof + safe exploration zones.",
                    "💬 Read 10 mins/day; ask questions about pictures.",
                    "🎶 Music + movement help mood and language.",
                    "🧩 Encourage independent play to build focus."
                ],
                "Preschool (3–5 yr)": [
                    "🎨 Let them help: tiny chores build competence.",
                    "📚 Daily storytime; let them retell the story.",
                    "👫 Practice sharing, turn-taking, and apologies.",
                    "🌳 30–60 mins outdoor play for sleep & mood.",
                    "🦷 Brush teeth AM/PM; dentist checkups."
                ],
                "Sleep Tips": [
                    "🌙 Consistent routine: bath → book → bed.",
                    "🧸 Night light & white noise if helpful.",
                    "📵 No screens 1 hour before bedtime.",
                    "🪄 Use a calm phrase: “It’s time to rest. I’m here.”",
                    "⏱️ Expect regressions during growth spurts."
                ],
                "Behavior & Emotions": [
                    "💖 Name feelings: “You’re mad because…”",
                    "🧘 Model calm breaths; kids mirror you.",
                    "⏳ Short time-ins/time-outs (1 min per year).",
                    "🎯 Praise specific behavior: “Great sharing!”",
                    "🔁 Repeat house rules simply and consistently."
                ],
            }

            guide_sel = st.selectbox("Select a guide:", list(guides_catalog.keys()), key="guide_sel_v2")
            st.markdown("### 📝 What to know")
            for tip in guides_catalog[guide_sel]:
                st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

        # ---------- TAB 2: NUTRITION ----------
        with tab2:
            nutrition_catalog = {
                "No restrictions": {
                    "Daily ideas": [
                        "🍚 Plate: ½ veggies, ¼ protein, ¼ whole grains.",
                        "🥛 2–3 servings dairy or fortified alternatives.",
                        "🍇 Fruit snacks; limit added sugar.",
                        "🌈 Colorful veggies = varied nutrients.",
                        "💧 Keep water within reach (small cup).",
                    ],
                    "Snack inspo": [
                        "🍎 Apple slices + peanut butter (thin spread).",
                        "🧀 Cheese + whole-grain crackers.",
                        "🍌 Banana + yogurt (or dairy-free).",
                        "🥕 Veggie sticks + hummus.",
                        "🌽 Sweet corn with a squeeze of lime."
                    ]
                },
                "Vegetarian": {
                    "Daily ideas": [
                        "🥦 Pair beans/lentils with grains for complete protein.",
                        "🥚 Eggs (if included) = B12 & choline for brain growth.",
                        "🥜 Nut butters (thinly spread) for healthy fats.",
                        "🍠 Iron from legumes + Vitamin C (citrus) for absorption.",
                        "🧂 Keep salt low; use herbs/spices for flavor."
                    ],
                    "Snack inspo": [
                        "🍞 Avocado toast fingers.",
                        "🥣 Oatmeal with chia & berries.",
                        "🥜 Apple + almond or sunflower butter.",
                        "🫘 Mini lentil patties.",
                        "🥛 Fortified soy or oat milk smoothie."
                    ]
                },
                "Allergy-aware": {
                    "Daily ideas": [
                        "🚫 Introduce one new food at a time; observe 3 days.",
                        "🍞 Gluten-free grains: rice, oats (GF), corn.",
                        "🥥 Dairy-free? Use fortified alternatives (check calcium).",
                        "🐟 Avoid cross-contamination: separate utensils/boards.",
                        "🩺 Keep an action plan & share with caregivers."
                    ],
                    "Snack inspo": [
                        "🍚 Rice cakes + mashed avocado.",
                        "🍇 Fresh fruit + dairy-free yogurt.",
                        "🥔 Baked potato wedges + olive oil.",
                        "🍗 (If not veg) Shredded chicken + steamed veg.",
                        "🍌 Banana oat cookies (no egg/dairy if needed)."
                    ]
                }
            }

            diet_sel = st.selectbox("Select diet type:", list(nutrition_catalog.keys()), key="diet_sel_v2")
            st.markdown("### 🍽️ Daily Meal Inspiration")
            for tip in nutrition_catalog[diet_sel]["Daily ideas"]:
                st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

            st.markdown("### 🧃 Easy Snack Ideas")
            for tip in nutrition_catalog[diet_sel]["Snack inspo"]:
                st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

        # ---------- TAB 3: SHOW ME HOW ----------
        with tab3:
            howto_catalog = {
                "Introduce Solid Foods": [
                    "👶 Start near 6 months; look for head control & interest.",
                    "🥄 First tastes: avocado, banana, sweet potato, peas.",
                    "🧪 One new food at a time; watch 2–3 days for reactions.",
                    "💧 Offer a few sips of water after solids.",
                    "🚫 Avoid honey, whole nuts, and cow’s milk under 1.",
                ],
                "Screen Time Balance": [
                    "📱 Under 2: avoid screens except video chats.",
                    "👩‍👧 Co-watch; talk about what you see together.",
                    "⏰ Timers help older kids (30–45 min blocks).",
                    "🎨 Offer swaps: coloring, blocks, music, outdoor time.",
                    "💤 No screens 1 hour before bed."
                ],
                "Positive Discipline": [
                    "💖 Catch them doing right; praise specific actions.",
                    "🧘 Kneel to eye level; calm voice first.",
                    "⏳ Use short, consistent time-outs (1 min per year).",
                    "🧭 Teach the alternative behavior you want.",
                    "🔁 Repeat simple rules; consistency beats intensity."
                ],
                "Toilet Training": [
                    "🚽 Introduce potty when they show readiness cues.",
                    "🧸 Make it positive; celebrate attempts.",
                    "🧻 Easy pull-down clothes help independence.",
                    "⏰ Try regular potty times after meals & before bath.",
                    "💧 Accidents happen—stay calm, reset, and try again."
                ]
            }

            how_sel = st.selectbox("Select a how-to:", list(howto_catalog.keys()), key="how_sel_v2")
            st.markdown("### 💡 Step-by-Step")
            for tip in howto_catalog[how_sel]:
                st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- JOURNAL (full width at bottom) --------------------
    from datetime import datetime

    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown("## 🪞 Parent Reflection Journal")

    # Initialize journal store
    if "journal_entries" not in st.session_state:
        st.session_state["journal_entries"] = []

    note = st.text_area(
        "Write one small thing you’re proud of today (anything counts 😊):",
        placeholder="E.g., I stayed calm during a tantrum. We read a book together. I took 3 deep breaths.",
        key="journal_note",
        height=120,
    )

    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("Save reflection", key="save_reflection_btn"):
            if note.strip():
                st.session_state["journal_entries"].append({
                    "ts": datetime.now().strftime("%b %d, %Y — %I:%M %p"),
                    "text": note.strip()
                })
                st.success("🌼 Saved. You’re doing great.")
            else:
                st.info("Write a few words before saving.")

    # Show previous entries (latest first)
    if st.session_state["journal_entries"]:
        st.markdown("### 📔 Your past reflections")
        for entry in reversed(st.session_state["journal_entries"][-10:]):
            st.markdown(
                f"<div class='tip-card'><div class='tip-header'>{entry['ts']}</div>{entry['text']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("<div class='muted'>No entries yet — your first note will appear here.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Page: Legal Rights ---
elif page == "legal":
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.header("⚖️ Safety & Rights")
    region = st.text_input("Enter your State / Pincode:", key="legal_reg")
    if region:
        ans = ask_chat(f"Legal rights in {region} for custody, child support, domestic violence, benefits.")
        # st.write(ans)
    st.markdown("### Key Topics")
    topics = ["Custody", "Child Support", "Domestic Violence Help", "Benefits Eligibility"]
    for t in topics:
        if st.button(t):
            st.write(ask_chat(f"For {region}, explain {t} rights."))
    
# --- Page: Community Finder ---
elif page == "community":
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.header("📍 Community")
    pin = st.text_input("Enter Pincode or City:", key="comm_pin")
    cat = st.selectbox("Category:", ["Childcare", "Pediatricians", "Food Bank", "Shelter","Medical Center"], key="comm_cat")
    if st.button("Search"):
        resp = ask_chat(f"Resources near {pin} for {cat}")
        st.write(resp)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Page: Postpartum Care ---
elif page == "postpartum":
    # --- Pretty slider & mood badge styles (scoped to this page) ---
    st.markdown("""
    <style>
    /* --- Mood Check-in Slider with dark panel --- */
    section.postpartum [data-testid="stWidgetLabel"] p {
      font-size: 1.2rem !important;
      font-weight: 600 !important;
      letter-spacing: .3px;
      color: #fff !important;
      margin-bottom: 0.8rem !important;
    }
    /* Outer container wrapper for slider background */
    section.postpartum div[data-testid="stSlider"] {
      background: rgba(0, 0, 0, 0.55) !important;
      border-radius: 18px !important;
      padding: 18px 24px !important;
      margin-bottom: 1rem !important;
      box-shadow: 0 8px 18px rgba(0,0,0,0.35) !important;
    }
    /* Red active track */
    section.postpartum .stSlider .rc-slider-track {
      height: 8px !important;
      border-radius: 999px !important;
      background: linear-gradient(90deg, #ff4d4d 0%, #ff7676 100%) !important;
    }
    /* Grey rail (inactive track) */
    section.postpartum .stSlider .rc-slider-rail {
      height: 8px !important;
      border-radius: 999px !important;
      background: rgba(255,255,255,0.25) !important;
    }
    /* Red glowing handle */
    section.postpartum .stSlider .rc-slider-handle {
      width: 22px !important;
      height: 22px !important;
      margin-top: -7px !important;
      border: 2px solid #fff !important;
      background: #ff4d4d !important;
      box-shadow: 0 0 8px rgba(255,80,80,0.55) !important;
    }
    section.postpartum .stSlider .rc-slider-handle:hover,
    section.postpartum .stSlider .rc-slider-handle:active {
      background: #ff7878 !important;
      box-shadow: 0 0 14px rgba(255,100,100,0.8) !important;
    }
    /* Hide numeric tooltip */
    section.postpartum .stSlider .rc-slider-tooltip { display: none !important; }
    /* Mood badge styling */
    .mood-badge {
      display: inline-flex; align-items: center; gap: 10px;
      padding: 10px 20px;
      background: rgba(0, 0, 0, 0.55);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 999px;
      font-size: 1.1rem; font-weight: 700; color: #fff;
      margin-top: 20px; margin-bottom: 20px; backdrop-filter: blur(6px);
    }
    </style>
    """, unsafe_allow_html=True)

    # Wrap this page so CSS targets only here
    st.markdown('<section class="postpartum">', unsafe_allow_html=True)

    st.header("🌼 Postpartum Care — Mind & Body")

    # ---------------- Mood Check-in ----------------
    st.subheader("Mood Check-in")
    mood = st.slider("How do you feel (1 = low … 5 = good)?", 1, 5, 3, key="mood")

    mood_map = {
        1: ("😞", "Hard day"),
        2: ("😕", "Low"),
        3: ("😌", "In-between"),
        4: ("🙂", "Okay"),
        5: ("🌟", "Great"),
    }
    emoji, label = mood_map[mood]
    st.markdown(f'<div class="mood-badge">{emoji} <span class="txt">{label}</span></div>', unsafe_allow_html=True)

    if st.button("Reflect"):
        if mood <= 2:
            st.write("Seems like a heavy day — here are 3 small steps:")
            st.write("- Breathe deeply 5 times\n- Walk for 2 minutes\n- Drink water")
        else:
            st.write("Thanks for checking in — keep going 💖")

    st.markdown("---")

    # ---------------- Mindfulness Corner (YouTube + Timer) ----------------
    import time
    from datetime import timedelta

    st.subheader("Mindfulness Corner")

    presets = {
        "2-min Box Breathing": "https://www.youtube.com/watch?v=uNeoLT1axSI&pp=ygUeMiBtaW4gYm94IGJyZWF0aGluZyBtZWRpdGF0aW9u",
        "3-min Body Scan": "https://www.youtube.com/watch?v=ihwcw_ofuME&pp=ygUaMyBtaW4gYm9keSBzY2FuIG1lZGl0YXRpb24%3D",
        "5-min Loving-Kindness": "https://www.youtube.com/watch?v=QMISP4M4GQo&pp=ygUgNSBtaW4gbG92aW5nIGtpbmRuZXNzIG1lZGl0YXRpb27SBwkJ_AkBhyohjO8%3D",
        "Calm Breathing": "https://www.youtube.com/watch?v=aNXKjGFUlMs",
        "Custom YouTube link…": ""
    }

    c_yt1, c_yt2 = st.columns([1.2, 1])
    with c_yt1:
        sel = st.selectbox("Pick a short guided practice:", list(presets.keys()), key="mind_yt_sel")
        url = presets[sel]
    with c_yt2:
        if sel == "Custom YouTube link…":
            url = st.text_input("Paste a YouTube URL:", placeholder="https://youtube.com/…", key="mind_yt_url")

    if url:
        st.video(url)
    else:
        st.caption("Tip: choose a practice or paste a YouTube link to play here.")

    st.markdown("---")

    st.markdown("**2-minute breathing timer** (choose any duration)")

    # Init session state for timer
    for k, v in {"timer_total": 120, "timer_remaining": 120, "timer_running": False}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1, 2])
    with c1:
        preset = st.selectbox(
            "Presets", ["1 min", "2 min", "3 min", "5 min", "10 min", "Custom"],
            index=1, help="Quick select a duration", key="timer_preset",
        )
    with c2:
        if preset != "Custom":
            minutes = int(preset.split()[0])
            custom_secs = st.number_input("Add seconds", 0, 59, 0, key="timer_extra")
            chosen_total = minutes * 60 + custom_secs
        else:
            chosen_total = st.number_input("Custom seconds", 10, 3600, 120, 10, key="timer_custom")
    with c3:
        if st.button("Set", use_container_width=True):
            st.session_state.timer_total = int(chosen_total)
            st.session_state.timer_remaining = int(chosen_total)
            st.session_state.timer_running = False
    with c4:
        st.caption("Tip: press **Set** after choosing a new duration.")

    disp = st.empty()
    def _fmt(sec: int) -> str:
        return str(timedelta(seconds=sec))[-5:] if sec < 3600 else str(timedelta(seconds=sec))

    disp.markdown(
        f"<div style='font-size:42px; font-weight:800; letter-spacing:1px; "
        f"padding:14px 18px; border-radius:16px; display:inline-block; "
        f"background:rgba(0,0,0,.45); color:#fff; border:1px solid rgba(255,255,255,.18);'>⏱️ {_fmt(st.session_state.timer_remaining)}</div>",
        unsafe_allow_html=True
    )

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state.timer_running = True
    with b2:
        if st.button("Pause", use_container_width=True):
            st.session_state.timer_running = False
    with b3:
        if st.button("Reset", use_container_width=True):
            st.session_state.timer_running = False
            st.session_state.timer_remaining = st.session_state.timer_total

    # Countdown (single-run loop, updates the placeholder)
# NOTE: Streamlit >=1.31 uses st.rerun() (experimental_rerun was removed)
    while st.session_state.timer_running and st.session_state.timer_remaining > 0:
        time.sleep(1)
        st.session_state.timer_remaining -= 1
        disp.markdown(
            f"<div style='font-size:42px; font-weight:800; letter-spacing:1px; "
            f"padding:14px 18px; border-radius:16px; display:inline-block; "
            f"background:rgba(0,0,0,.45); color:#fff; border:1px solid rgba(255,255,255,.18);'>⏱️ {_fmt(st.session_state.timer_remaining)}</div>",
            unsafe_allow_html=True
        )
        st.rerun()  # <-- updated call


    # Finish state
    if st.session_state.timer_remaining == 0 and st.session_state.timer_running:
        st.session_state.timer_running = False
        st.balloons()
        st.success("Nice work — timer complete! 🌿")

    st.markdown("---")

    # ---------------- Body Recovery & SOS ----------------
    st.subheader("Body Recovery")

    recovery_guides = {
        "💤 Gentle Stretch": {
            "desc": "Start with light neck, shoulder, and spine movements to ease stiffness. Focus on posture alignment and slow breathing.",
            "steps": [
                "🧘‍♀️ Neck rolls — 5 each direction",
                "🙆‍♀️ Shoulder shrugs — 10 reps",
                "🦋 Seated butterfly stretch — 30 seconds hold",
                "🌬️ Deep belly breathing — 5 rounds"
            ],
            "resource": "https://www.youtube.com/watch?v=s5ly73hyRhg&pp=ygUdZ2VudGxlIHN0cmV0Y2ggZm9yIHBvc3RwYXJ0dW0%3D"
        },
        "🌸 Pelvic Floor Basics": {
            "desc": "Strengthen pelvic muscles to improve bladder control and healing. You can do this sitting or lying down.",
            "steps": [
                "Contract pelvic muscles (like stopping urine flow)",
                "Hold for 3–5 seconds, then release for 5 seconds",
                "Repeat 10 times, 2–3 sets per day",
                "Avoid holding your breath — stay relaxed"
            ],
            "resource": "https://www.youtube.com/watch?v=-hSZqmuN41E&pp=ygUlcGVsdmljIGZsb29yIGV4ZXJjaXNlcyBmb3IgcG9zdHBhcnR1bQ%3D%3D"
        },
        "🤱 Core Reconnection (Diastasis-Safe)": {
            "desc": "Helps reconnect abdominal muscles safely after delivery.",
            "steps": [
                "Lie on your back, knees bent, feet flat",
                "Exhale and pull your belly button gently toward your spine",
                "Inhale to relax — keep ribs soft",
                "Add gentle arm lifts for progress"
            ],
            "resource": "https://www.youtube.com/watch?v=aT42-duXvMg&pp=ygUdY29yZSBleGVyY2lzZXMgZm9yIHBvc3RwYXJ0dW0%3D"
        },
        "🧘‍♀️ Postnatal Yoga Flow": {
            "desc": "A calming yoga sequence designed for postpartum recovery and relaxation.",
            "steps": [
                "Cat-cow stretch for spine mobility",
                "Low lunge hip opener (hold 30s each side)",
                "Bridge pose (gentle core activation)",
                "Rest in child’s pose with slow breathing"
            ],
            "resource": "https://www.youtube.com/watch?v=hpQH-mbd07Y&pp=ygUTcG9zdG5hdGFsIHlvZ2EgZmxvdw%3D%3D"
        },
        "🏃‍♀️ Walk & Breathe Routine": {
            "desc": "Low-impact walk and breath synchronization to boost mood and circulation.",
            "steps": [
                "Walk at an easy pace for 5–10 minutes",
                "Focus on breathing: inhale 3 steps, exhale 3 steps",
                "Add arm swings to improve rhythm",
                "Cool down: slow breathing and shoulder rolls"
            ],
            "resource": "https://www.youtube.com/watch?v=uC9fHDyH4-E&pp=ygUacG9zdG5hdGFsIHdhbGsgYW5kIGJyZWF0aGU%3D"
        },
        "🪷 Postpartum Relaxation Meditation": {
            "desc": "A short guided meditation to ease anxiety and promote self-compassion.",
            "steps": [
                "Find a quiet spot and sit or lie comfortably",
                "Close your eyes and breathe deeply",
                "Repeat: 'I am healing. I am patient. I am enough.'",
                "Listen to a guided session below"
            ],
            "resource": "https://www.youtube.com/watch?v=bXk916CpGxw&pp=ygUZbWVkaXRhdGlvbiBmb3IgcG9zdHBhcnR1bQ%3D%3D"
        },
    }

    sel_rec = st.selectbox("Choose recovery:", list(recovery_guides.keys()), key="rec_sel")

    guide = recovery_guides[sel_rec]
    st.markdown(f"### 🩰 {sel_rec}")
    st.write(guide["desc"])
    st.markdown("#### 🪄 Steps:")
    for step in guide["steps"]:
        st.markdown(f"- {step}")

    st.markdown("#### 🎥 Watch / Follow Along:")
    st.video(guide["resource"])

    st.markdown("---")
    st.markdown("**SOS Helplines**")
    st.markdown("- 🧡 Domestic Abuse: 800-799-7233")
    st.markdown("- 💬 Maternal Mental Health: 1-833-825-6262")
    st.markdown("- 🌐 [Postpartum Support International](https://www.postpartum.net/) – 24/7 helpline: 1-800-944-4773")

    st.markdown("</section>", unsafe_allow_html=True)

# --- Page: Financial Wellness ---
elif page == "finance":
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.header("💵 Financial Wellness")
    with st.form("budget_form"):
        income = st.number_input("Monthly Income ($):", min_value=0.0, step=50.0, key="f_inc")
        essentials = st.number_input("Essentials (rent, food):", min_value=0.0, step=50.0, key="f_ess")
        savings = st.number_input("Savings goal:", min_value=0.0, step=50.0, key="f_sav")
        sub = st.form_submit_button("Analyze")
        if sub:
            remainder = income - essentials - savings
            st.write(f"You have **${remainder:.2f}** left this month.")
            st.bar_chart({"Essentials": essentials, "Savings": savings, "Leftover": remainder})
            st.markdown("### Micro-savings ideas")
            st.write("- Brew coffee at home\n- Pack lunch\n- Use free resources/library")

    st.markdown("---")
    st.subheader("Aid / Benefit Suggestions")
    region = st.text_input("State / Region:", key="fin_reg")
    if region:
        st.write(ask_chat(f"Financial aid in {region}"))
    st.markdown("</div>", unsafe_allow_html=True)


# --- Page: Age‑Smart Coach ---
# --- Page: Smart Meal Helper ---
elif page == "meals":
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.header("🥗 Smart Meal Planner")
    st.caption("Simple inputs → curated meal ideas from your local CSV (no LLM).")

    # ---- Filters row ----
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])

    with c1:
        age_mo = st.number_input("Child age (months)", min_value=6, max_value=240, value=24, step=1)
    with c2:
        diet = st.selectbox("Preference", ["veg", "egg", "chicken", "any"], index=0)
    with c3:
        mood = st.selectbox("Mood / Energy", ["Any", "Low energy", "In-between", "Active"], index=1)
    with c4:
        meal_type = st.multiselect("Meal type", ["breakfast", "lunch", "snack", "dinner"], default=["breakfast","lunch","snack","dinner"])

    c5, c6 = st.columns([1.2, 1.8])
    with c5:
        max_prep = st.slider("Max prep time (min)", 5, 45, 25, step=5)
    with c6:
        sort_by = st.selectbox("Sort by", ["Best match", "Highest protein", "Lowest prep time", "Calories ↑", "Calories ↓"])

    if MEALS_DF.empty:
        st.warning("`static_content/meals.csv` not found or empty. Add the CSV I shared and rerun.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ---- Filtering logic ----
    df = MEALS_DF.copy()

    # Age window
    df = df[(df["age_min"] <= age_mo) & (df["age_max"] >= age_mo)]

    # Diet
    if diet != "any":
        df = df[df["diet"].str.lower() == diet]

    # Meal type
    if meal_type:
        df = df[df["meal_type"].str.lower().isin([m.lower() for m in meal_type])]

    # Prep time
    df = df[df["prep_mins"] <= max_prep]

    # Mood → Tag heuristics
    mood_map = {
        "Any": [],
        "Low energy": ["soft", "comfort", "soothing", "light"],
        "In-between": ["comfort", "wholegrains", "energy"],
        "Active": ["protein", "energy", "wholegrains"]
    }
    wanted_tags = set(mood_map.get(mood, []))
    if wanted_tags:
        # score by number of tag hits
        def _score_tags(x):
            tags = set([t.strip().lower() for t in x.split("|")]) if "|" in x else set([t.strip().lower() for t in x.split(",")])
            return len(tags & wanted_tags)
        df["tag_score"] = df["tags"].apply(_score_tags)
    else:
        df["tag_score"] = 0

    # Sorting
    if sort_by == "Highest protein":
        df = df.sort_values(["protein_g", "tag_score"], ascending=[False, False])
    elif sort_by == "Lowest prep time":
        df = df.sort_values(["prep_mins", "tag_score"], ascending=[True, False])
    elif sort_by == "Calories ↑":
        df = df.sort_values(["cals", "tag_score"], ascending=[True, False])
    elif sort_by == "Calories ↓":
        df = df.sort_values(["cals", "tag_score"], ascending=[False, False])
    else:  # Best match (tag score → protein density → prep)
        df = df.sort_values(["tag_score", "protein_density", "prep_mins"], ascending=[False, False, True])

    # ---- Energy tip (optional boosters.csv) ----
    tip_html = ""
    if not BOOSTERS_DF.empty:
        # filter by diet & age if possible
        bdf = BOOSTERS_DF.copy()
        if diet != "any":
            bdf = bdf[(bdf["diet"].str.lower().isin([diet, "any"]))]
        bdf = bdf[(bdf["age_min"] <= age_mo) & (bdf["age_max"] >= age_mo)]
        if bdf.empty:  # fallback to any
            bdf = BOOSTERS_DF.copy()
        row = bdf.sample(1).iloc[0]
        tip_html = f"""<div class="energy-tip">⚡ Energy tip: <b>{row['booster']}</b> — {row['benefit']}. <span style="opacity:.9">{row['tip']}</span></div>"""

    # ---- Results ----
    st.markdown("---")
    left_count, right_actions = st.columns([1,1])
    with left_count:
        st.subheader(f"Results · {len(df)} meals")
        if tip_html:
            st.markdown(tip_html, unsafe_allow_html=True)

    with right_actions:
        csv_btn = st.download_button(
            "Download filtered CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="meal_suggestions.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Toggle between cards and table
    view_mode = st.toggle("Show as table", value=False)
    if view_mode:
        show_cols = ["title","meal_type","diet","cals","protein_g","prep_mins","notes","recipe"]
        st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, height=420)
    else:
        # render pretty cards (top 12)
        show_df = df.head(12).reset_index(drop=True)
        for _, r in show_df.iterrows():
            badges = [
                f"<span>{r['meal_type'].title()}</span>",
                f"<span>{r['diet'].title()}</span>",
                f"<span>⏱ {int(r['prep_mins'])}m</span>",
                f"<span>🔥 {int(r['cals'])} kcal</span>",
                f"<span>💪 {int(r['protein_g'])} g</span>",
            ]
            st.markdown(
                f"""
                <div class="meal-card">
                  <h4>{r['title']}</h4>
                  <div class="meal-meta">{r['notes']}</div>
                  <div class="meal-badges">{' '.join(badges)}</div>
                  <div class="meal-notes"><b>Quick recipe:</b> {r['recipe']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# --- MOM'S NOURISHMENT SECTION ---

    st.markdown("### 🌿 For Mom — Nourish Yourself Too")

    # 1️⃣ Inputs
    mood = st.selectbox("How are you feeling today?", ["Tired", "Stressed", "Okay", "Energetic"], key="mom_mood")
    time_avail = st.selectbox("How much time do you have?", ["<10 min", "10–20 min", "30+ min"], key="mom_time")
    diet_m = st.selectbox("Your diet type:", ["Vegetarian", "Egg", "Chicken"], key="mom_diet")

    # 2️⃣ Load mom meals CSV
    try:
        mom_meals_df = pd.read_csv("static_content/mom_meals.csv")
    except FileNotFoundError:
        st.error("⚠️ Missing file: static_content/mom_meals.csv — please add it to your project folder.")
        mom_meals_df = pd.DataFrame(columns=["title", "diet", "mood", "time", "benefit", "tip"])

    # 3️⃣ Filter logic
    results = mom_meals_df[
        (mom_meals_df["diet"].str.lower() == diet_m.lower())
        & (mom_meals_df["time"].str.contains(time_avail))
        & (mom_meals_df["mood"].str.contains(mood))
    ]

    # 4️⃣ Display results in pretty cards
    if not results.empty:
        for _, row in results.iterrows():
            st.markdown(f"""
            <div style='background: rgba(0,0,0,0.55); border-radius: 18px; padding: 16px 20px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.25); color: #fff;'>
            <h4 style='margin:0;'>{row['title']}</h4>
            <p style='margin:4px 0 0 0; font-size:0.9rem; opacity:.9;'>💪 {row['benefit']}</p>
            <p style='margin:2px 0 0 0; font-size:0.9rem; opacity:.85;'>✨ Tip: {row['tip']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No match found — try a different mood or time range!")

    st.caption("💖 Remember: You deserve to eat well, too — even small bites count!")

