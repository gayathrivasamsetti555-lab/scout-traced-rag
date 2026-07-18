import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
    --paper:#f4efe2; --paper-card:#faf6ec; --ink:#20211c; --ink-soft:#5b5a4f;
    --rule:#d9d0b8; --rust:#c1502d; --sage:#69765c;
}
.stApp{background:var(--paper); color:var(--ink); font-family:'Inter',sans-serif;}
.scout-mast{border-bottom:2px solid var(--ink); padding-bottom:14px; margin-bottom:8px;}
.scout-mast h1{font-family:'Fraunces',serif; font-size:34px; font-weight:600; margin:0;}
.scout-tagline{font-family:'Fraunces',serif; font-style:italic; font-size:14.5px; color:var(--ink-soft);}
.scout-dateline{font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--ink-soft);
    border-top:1px solid var(--rule); margin-top:12px; padding-top:8px; display:flex; gap:20px;}

.panel-label{font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--rust);
    text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; display:block;}

.answer-card{background:var(--paper-card); border:1px solid var(--rule); border-radius:4px; padding:24px 26px;}
.answer-card p{font-family:'Fraunces',serif; font-size:18px; line-height:1.65; color:var(--ink);}
.answer-note{margin-top:16px; padding-top:12px; border-top:1px dashed var(--rule);
    font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--ink-soft); line-height:1.6;}

.passage-card{background:var(--paper-card); border:1px solid var(--rule); border-radius:4px;
    padding:16px 18px; margin-bottom:14px; position:relative;}
.passage-card.top{border-color:var(--rust);}
.passage-card.top::before{content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:var(--rust);}
.passage-title{font-family:'Fraunces',serif; font-size:15.5px; font-weight:600;}
.passage-section{font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-soft); text-transform:uppercase;}
.passage-text{font-size:13.5px; color:var(--ink-soft); margin:8px 0 10px; line-height:1.55;}
.score-num{font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--ink-soft);}

div.stButton > button{
    background:var(--ink); color:var(--paper-card); border:none; border-radius:4px;
    font-family:'JetBrains Mono',monospace; font-size:13px;
}
div.stButton > button:hover{background:var(--rust); color:var(--paper-card);}

.board-card{background:var(--paper-card); border:1px solid var(--rule); border-radius:4px; padding:20px 22px;}
.board-note{font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--ink-soft);
    margin-top:10px; padding-top:10px; border-top:1px dashed var(--rule); line-height:1.6;}

.rec-card{background:var(--paper-card); border:1px solid var(--rule); border-radius:4px;
    padding:14px 16px; margin-bottom:12px;}
.rec-tag{font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--sage);
    text-transform:uppercase; letter-spacing:.08em;}
.rec-title{font-family:'Fraunces',serif; font-size:14.5px; font-weight:600; margin:4px 0 6px;}
.rec-text{font-size:12.5px; color:var(--ink-soft); line-height:1.5;}

.section-head{display:flex; align-items:center; gap:10px; margin:28px 0 12px;}
.section-num{font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--paper-card);
    background:var(--ink); border-radius:50%; width:22px; height:22px; display:flex;
    align-items:center; justify-content:center; flex-shrink:0;}
.section-head h3{font-family:'Fraunces',serif; font-size:17px; font-weight:600; margin:0;}

.quality-badge{display:inline-flex; align-items:center; gap:10px; background:var(--paper-card);
    border:1px solid var(--rule); border-radius:20px; padding:6px 16px 6px 6px; margin-bottom:14px;}
.quality-badge .num{font-family:'Fraunces',serif; font-size:20px; font-weight:700;
    background:var(--ink); color:var(--paper-card); border-radius:50%; width:38px; height:38px;
    display:flex; align-items:center; justify-content:center;}
.quality-badge .lbl{font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--ink-soft);
    text-transform:uppercase; letter-spacing:.06em;}

.rationale-box{font-family:'Fraunces',serif; font-style:italic; font-size:14.5px;
    color:var(--ink); background:var(--paper-card); border-left:3px solid var(--sage);
    border-radius:0 4px 4px 0; padding:12px 16px; margin-bottom:18px; line-height:1.6;}

.flow-wrap{margin:14px 0 20px;}

.sidebar-rec-card{background:var(--paper-card); border:1px solid var(--rule); border-radius:4px;
    padding:10px 12px; margin-bottom:10px;}
.sidebar-rec-tag{font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--sage);
    text-transform:uppercase; letter-spacing:.06em;}
.sidebar-rec-title{font-family:'Fraunces',serif; font-size:13px; font-weight:600; margin:3px 0;}

.source-meta{margin-bottom:2px; text-transform:none;}
.source-title{font-family:'Fraunces',serif; font-size:13.5px; font-weight:600; color:var(--ink);
    text-transform:none; letter-spacing:0;}
.source-byline{font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--ink-soft);
    text-transform:none; letter-spacing:0; margin-top:2px;}
.source-raw{font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-soft);
    text-transform:uppercase;}

mark.hl-sentence{background:#f0d9a8; color:var(--ink); padding:1px 3px; border-radius:2px;
    font-weight:600; box-decoration-break:clone;}

.stars{font-size:12px; color:var(--rust); letter-spacing:1px;}
</style>
"""


def apply_styles():
    st.markdown(CSS, unsafe_allow_html=True)