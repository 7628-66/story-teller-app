# ruff: noqa: I001
import streamlit as st
from story_app import StoryConfig, StoryGenerator

st.set_page_config(page_title="StoryGen", page_icon="📖", layout="centered")

st.title("StoryGen 📖")
st.caption("Offline-friendly story generator. Toggle arc, tone, and world; generate instantly.")

with st.sidebar:
    st.header("Configuration")
    genre = st.selectbox("Genre", ["cozy mystery", "low fantasy noir", "science fiction", "sci-fantasy", "speculative fiction"], index=0)
    tone = st.selectbox("Tone", ["hopeful", "whimsical", "gritty", "noir"], index=1)
    setting = st.text_input("Setting", value="quaint English village")
    pov = st.selectbox("POV", ["first", "third"], index=1)
    tense = st.selectbox("Tense", ["past", "present"], index=0)
    ending = st.selectbox("Ending", ["bittersweet", "triumphant", "open-ended"], index=0)
    length = st.selectbox("Length", ["flash", "short", "novelette", "chapter"], index=0)
    themes = st.text_input("Themes (comma)", value="redemption")
    elements = st.text_input("Elements (comma)", value="lockbox")
    arc_style = st.selectbox("Arc Style", ["rise-fall", "fall-rise", "steady climb"], index=1)
    protagonist_role = st.text_input("Protagonist Role", value="outsider")
    protagonist_age = st.selectbox("Protagonist Age", ["teen", "adult", "elder"], index=0)
    time_period = st.selectbox("Time Period", ["medieval", "modern", "futuristic"], index=1)
    tech_level = st.selectbox("Tech Level", ["primitive", "industrial", "futuristic"], index=1)
    magic_system = st.text_input("Magic System", value="forbidden")
    weather_mood = st.text_input("Weather Mood", value="stormy")
    cultural_conflict = st.text_input("Cultural Conflict", value="tradition vs progress")
    deterministic = st.checkbox("Deterministic", value=True)

cfg = StoryConfig(
    genre=genre,
    tone=tone,
    pov=pov,
    tense=tense,
    length=length,
    ending=ending,
    themes=[t.strip() for t in themes.split(',') if t.strip()],
    required_elements=[e.strip() for e in elements.split(',') if e.strip()],
    setting=setting,
    arc_style=arc_style,
    protagonist_age=protagonist_age,
    protagonist_role=protagonist_role,
    time_period=time_period,
    tech_level=tech_level,
    magic_system=magic_system,
    weather_mood=weather_mood,
    cultural_conflict=cultural_conflict,
)

# Generator defaults to offline (no model)
sg = StoryGenerator(cfg, load_model=False, deterministic=deterministic, dry_run=False)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Generate Hook"):
        st.code(sg.generate_hook())
with col2:
    if st.button("Generate Outline"):
        outline = sg.generate_outline()
        st.write(outline)
with col3:
    if st.button("Generate Story"):
        st.write(sg.generate_story())

st.divider()
st.subheader("Choose-Your-Own Midpoint")
mid_col1, mid_col2 = st.columns(2)
with mid_col1:
    if st.button("Trust"):
        sg.set_branch_override('midpoint.choice', 'y')
        st.write(sg.generate_midpoint())
with mid_col2:
    if st.button("Distrust"):
        sg.set_branch_override('midpoint.choice', 'n')
        st.write(sg.generate_midpoint())

st.divider()
# Export/download text
export_choice = st.selectbox("Export format", [".txt", ".md"], index=1)
if st.button("Export Story"):
    story = sg.generate_story()
    if export_choice == ".md":
        st.download_button("Download story.md", data=story, file_name="story.md", mime="text/markdown")
    else:
        st.download_button("Download story.txt", data=story, file_name="story.txt", mime="text/plain")
