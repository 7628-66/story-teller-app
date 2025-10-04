# StoryGen

Offline-friendly, AI-assisted story generator with a friendly CLI and Streamlit web app.

## Who is this for?
- Writers: a free brainstorming buddy to spark hooks, outlines, and drafts
- Teachers: a creative writing tool with branching choices for class activities
- RPG GMs: quick plot seeds, NPC backstories, and world notes

## Install
- One-line user install:

```
pip install storygen
```

- Dev install from this repo:

```
pip install -e .
```

## CLI
- Run anywhere after install:

```
storygen --yes --no-model --output story --length short --themes redemption --elements lockbox --genre "cozy mystery" --tone whimsical --setting "quaint English village" --deterministic --arc-style fall-rise --weather-mood stormy
```

- Interactive wizard:

```
storygen --wizard
```

### Exports
- Save to a file using `--save output.md` or `output.txt`.
- If you install extras, you can export `.docx` and `.epub` by providing those extensions.

Extras install:

```
pip install storygen[export]
```

## Streamlit Web App
Launch the web UI:

```
pip install storygen[ui]
streamlit run streamlit_app.py
```

Features:
- Dropdowns for Genre, Tone, Arc Style, etc.
- Generate Hook/Outline/Story buttons
- Choose-Your-Own-Adventure midpoint buttons (Trust/Distrust)
- Download `.txt` or `.md`

## Docker
Optional container for zero Python setup:

```
# Build locally
docker build -t yourname/storygen .

# Run CLI
docker run -it yourname/storygen storygen --yes --no-model --output story

# Run Streamlit app (maps port 8501)
docker run -p 8501:8501 yourname/storygen streamlit run streamlit_app.py
```

## Examples
See `examples/` for ready-to-run configs:
- `fantasy_config.json`
- `cyberpunk_config.json`

## Starter data
The repo ships minimal `templates.json` and `worlds.json` so you can plug-and-play.

## Contributing
- Add settings, templates, and starter packs via JSON.
- Share screenshots/GIFs of the Streamlit app.

## License
MIT