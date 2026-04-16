import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import joblib
import streamlit as st
import plotly.graph_objects as go
from llama_api_final import FredLLMAgent
from gpt_api_final import OpenAIFredAgent

# constants
MODEL_ROLE = "ai"
AI_AVATAR_ICON = "✨"
HISTORY_DIR = "history"

os.makedirs(HISTORY_DIR, exist_ok=True)

# initialize session_state
if "agent" not in st.session_state:
    st.session_state.agent = FredLLMAgent(model="llama3.2", verbose=True, few_shot=True)
    # st.session_state.agent = OpenAIFredAgent(verbose=True)
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(time.time())
if "chat_title" not in st.session_state:
    st.session_state.chat_title = f"ChatSession-{st.session_state.chat_id}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chart_data" not in st.session_state:
    st.session_state.chart_data = []

# chat list
try:
    past_chats: dict = joblib.load(f"{HISTORY_DIR}/past_chats_list")
except:
    past_chats = {}

# side bar for previous chat selection
with st.sidebar:
    st.write("# Past Chats")
    selected = st.selectbox(
        label="Pick a past chat",
        options=["__new__"] + list(past_chats.keys()),
        format_func=lambda x: "New Chat" if x == "__new__" else past_chats.get(x, x),
    )
    if selected == "__new__":
        if st.session_state.chat_id in past_chats:
            st.session_state.chat_id    = str(time.time())
            st.session_state.chat_title = f"ChatSession-{st.session_state.chat_id}"
            st.session_state.messages   = []
            st.session_state.chart_data = []
    else:
        if selected != st.session_state.chat_id:
            st.session_state.chat_id    = selected
            st.session_state.chat_title = past_chats[selected]
            st.session_state.chart_data = []
            try:
                st.session_state.messages = joblib.load(f"{HISTORY_DIR}/{selected}-st_messages")
            except:
                st.session_state.messages = []

if not st.session_state.messages:
    try:
        st.session_state.messages = joblib.load(
            f"{HISTORY_DIR}/{st.session_state.chat_id}-st_messages"
        )
    except:
        pass


def extract_chart_series(api_results: list) -> list:
    """extract data for creating chart from api call results"""
    series_list = []
    for r in api_results:
        if not r.get("success"):
            continue
        dates, values = [], []

        # extract data from raw_observations
        for obs in r.get("raw_observations") or r.get("data", []):
            try:
                dates.append(obs["date"])
                values.append(float(obs["value"]))
            except (KeyError, ValueError):
                continue

        # analysis.full_timeseries
        # if not dates:
        #     for point in r.get("analysis", {}).get("full_timeseries", []):
        #         try:
        #             dates.append(point["date"])
        #             values.append(float(point["value"]))
        #         except (KeyError, ValueError):
        #             continue

        if dates and values:
            series_list.append({
                "series_id":      r.get("series_id", ""),
                "indicator_name": r.get("indicator_name") or r.get("series_id", ""),
                "units":          r.get("units", ""),
                "dates":          dates,
                "values":         values,
            })
    return series_list


def render_chart(chart_data: list):
    """generate the chart"""
    if not chart_data:
        return
    fig = go.Figure()
    for s in chart_data:
        fig.add_trace(go.Scatter(
            x=s["dates"],
            y=s["values"],
            mode="lines",
            name=f"{s['series_id']} — {s['indicator_name']}",
            hovertemplate="%{x}<br>%{y:.2f} " + (s["units"] or "") + "<extra></extra>",
        ))
    y_label = chart_data[0]["units"] if len(chart_data) == 1 else "Value"
    fig.update_layout(
        margin=dict(t=40, b=30, l=10, r=10),
        height=300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title=y_label, showgrid=True, gridcolor="#e0e0e0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# message bubble in scrollable container (HTML)
def messages_to_html(messages: list) -> str:
    AVATAR_USER = "🧑‍💻"
    AVATAR_AI   = "✨"
    AVATAR_STYLE = (
        "width:36px; height:36px; border-radius:50%; background:#e8e8e8;"
        "display:flex; align-items:center; justify-content:center;"
        "font-size:18px; flex-shrink:0;"
    )

    bubbles = []
    for m in messages:
        role    = m["role"]
        content = m["content"].replace("\n", "<br>")

        if role == "user":
            bubble = f"""
            <div style="display:flex; justify-content:flex-end; align-items:flex-end; gap:8px; margin:10px 0;">
              <div style="background:#0084ff; color:white; padding:10px 14px;
                          border-radius:18px 18px 4px 18px; max-width:72%;
                          font-size:14px; line-height:1.6;">
                {content}
              </div>
              <div style="{AVATAR_STYLE}">{AVATAR_USER}</div>
            </div>"""
        else:
            bubble = f"""
            <div style="display:flex; justify-content:flex-start; align-items:flex-end; gap:8px; margin:10px 0;">
              <div style="{AVATAR_STYLE} background:#ede9fe;">{AVATAR_AI}</div>
              <div style="background:#f0f0f0; color:#111; padding:10px 14px;
                          border-radius:18px 18px 18px 4px; max-width:72%;
                          font-size:14px; line-height:1.6;">
                {content}
              </div>
            </div>"""
        bubbles.append(bubble)
    return "".join(bubbles)


# CSS：fixed chart + scrollable chat container + fixed input bar
st.markdown("""
<style>
/* hide Streamlit default page edges */
.block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }

/* scrollable chat container */
#chat-history {
    height: 750px;
    overflow-y: auto;
    padding: 12px 16px;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 8px;
}

/* fixed input bar */
#chat-history { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)


# main page layout
st.write("# Chat with FRED Agent")

# scrollable chat history
chat_html = messages_to_html(st.session_state.messages)
# JS let container automatically scroll down to the bottom when new chat is created 
scroll_js = """
<script>
  const el = document.getElementById('chat-history');
  if (el) el.scrollTop = el.scrollHeight;
</script>
"""
st.markdown(
    f'<div id="chat-history">{chat_html}</div>{scroll_js}',
    unsafe_allow_html=True
)

# input bar
if prompt := st.chat_input("Ask me about economic data..."):

    if st.session_state.chat_id not in past_chats:
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, f"{HISTORY_DIR}/past_chats_list")

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        result = st.session_state.agent.process_question(prompt)

    if result.get("success"):
        full_response = result["final_answer"]
    else:
        full_response = f"⚠️ Error: {result.get('error', 'Unknown error')}"

    st.session_state.messages.append(
        {"role": MODEL_ROLE, "content": full_response, "avatar": AI_AVATAR_ICON}
    )
    joblib.dump(
        st.session_state.messages,
        f"{HISTORY_DIR}/{st.session_state.chat_id}-st_messages",
    )
    st.rerun()