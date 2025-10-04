# StoryGen 📚✨

An offline-friendly, AI-assisted story generator with CLI and Streamlit web interfaces. Generate complete stories with customizable themes, genres, branching narratives, and rich world-building elements.

## 🎯 Who is this for?
- **Writers**: A free brainstorming buddy to spark hooks, outlines, and complete drafts
- **Teachers**: A creative writing tool with branching choices for classroom activities  
- **RPG Game Masters**: Quick plot seeds, NPC backstories, and rich world-building notes
- **Students**: Learn story structure, narrative arcs, and creative writing techniques

## ✨ Features

### Core Capabilities
- **Offline-First**: Works without internet or AI models (deterministic generation)
- **Optional AI Enhancement**: Use `--use-model` to enable transformer-based text generation
- **Multiple Interfaces**: CLI wizard, programmatic CLI, and Streamlit web app
- **Rich Story Elements**: Themes, genres, character archetypes, world-building variables
- **Branching Narratives**: Interactive midpoint choices (Trust/Distrust paths)
- **Export Formats**: `.txt`, `.md`, `.docx`, `.epub` (with optional dependencies)

### Story Customization
- **Narrative Structure**: Choose from multiple story arc styles (fall-rise, hero's journey, etc.)
- **World Building**: Time periods, tech levels, magic systems, cultural conflicts
- **Character Development**: Protagonist age/role, character motivations
- **Atmospheric Elements**: Weather moods, settings, tones
- **Genre Variety**: Mystery, fantasy, sci-fi, literary fiction, and more

## 🚀 Quick Start

### Installation
```bash
# One-line user install
pip install storygen

# Or install with all optional features
pip install storygen[export,ui]

# Dev install from source
git clone https://github.com/yourusername/story-generator.git
cd story-generator
pip install -e .[dev]
```

### Basic Usage
```bash
# Interactive wizard (recommended for first-time users)
storygen --wizard

# Quick story generation (offline)
storygen --yes --output story --length short --themes redemption --elements lockbox --genre "cozy mystery"

# Generate with AI enhancement
storygen --use-model --yes --output story --length novella --themes "coming of age" --genre fantasy
```

## 📖 Interfaces

### 1. CLI Wizard (Interactive)
```bash
storygen --wizard
```
- Guided prompts for all story elements
- Colorized menus (when Rich is available)
- Live preview before generation
- Export options at the end

### 2. Programmatic CLI
```bash
storygen \
  --yes \
  --output story \
  --length short \
  --themes redemption \
  --elements lockbox \
  --genre "cozy mystery" \
  --tone whimsical \
  --setting "quaint English village" \
  --arc-style fall-rise \
  --protagonist-age teen \
  --weather-mood stormy \
  --save output.md
```

### 3. Streamlit Web App
```bash
pip install storygen[ui]
streamlit run streamlit_app.py
```

**Features:**
- Dropdown menus for all story parameters
- Generate Hook/Outline/Story buttons  
- Interactive midpoint choices (Trust/Distrust branching)
- Download stories as `.txt` or `.md` files
- Real-time preview with syntax highlighting

## 📁 Export Formats

StoryGen supports multiple export formats with graceful fallbacks:

### Core Formats (Always Available)
- **`.txt`**: Plain text format
- **`.md`**: Markdown with proper heading structure

### Optional Formats (Requires `pip install storygen[export]`)
- **`.docx`**: Microsoft Word document (via python-docx)
- **`.epub`**: E-book format (via ebooklib)

```bash
# Export examples
storygen --save my_story.md      # Markdown
storygen --save my_story.docx    # Word document
storygen --save my_story.epub    # E-book
```

## 🐳 Docker Support

Zero Python setup required:

```bash
# Build locally
docker build -t storygen .

# Run CLI
docker run -it storygen storygen --wizard

# Run Streamlit app (accessible at http://localhost:8501)
docker run -p 8501:8501 storygen streamlit run streamlit_app.py --server.address 0.0.0.0
```

## 📚 Examples & Templates

The repository includes starter configurations and templates:

### Example Configurations
- `examples/fantasy_config.json` - Medieval fantasy with magic systems
- `examples/cyberpunk_config.json` - Futuristic dystopian narratives

### Starter Data Files
- `templates.json` - Story structure templates for different genres
- `worlds.json` - Pre-built world-building elements and settings

```bash
# Use example configuration
storygen --config examples/fantasy_config.json --yes
```

## 🔧 Development Journey & Lessons Learned

This project evolved through several phases, each teaching valuable lessons about software development:

### Phase 1: Core Story Generation
**Goal**: Basic story generation with customizable parameters
**Challenges**:
- Originally tried complex AI integration first (mistake!)
- Learned to start simple: deterministic generation with templates
- **Key Lesson**: Build offline-first, add AI as enhancement

### Phase 2: Packaging & Distribution  
**Goal**: Make it easy to install and use
**Challenges**:
- Initial packaging errors with missing entry points
- Dependency conflicts between optional and required packages
- **Key Lessons**: 
  - Use `pyproject.toml` with proper `[project.scripts]` entry points
  - Implement optional dependencies with extras (`storygen[export,ui]`)
  - Default to offline mode to lower barrier to entry

### Phase 3: User Experience
**Goal**: Multiple interfaces for different user preferences
**Challenges**:
- CLI-only was intimidating for non-technical users
- Export functionality initially fragile with hard dependencies
- **Key Lessons**:
  - Provide both wizard (interactive) and programmatic CLI modes
  - Add web UI with Streamlit for broader accessibility
  - Implement graceful fallbacks for optional features

### Phase 4: Quality & Reliability
**Goal**: Professional-grade code quality and testing
**Challenges Found & Fixed**:

#### Linting Issues (Ruff)
```python
# ❌ Problem: Import ordering inconsistencies
from story_app import StoryConfig
import sys
import json

# ✅ Solution: Proper import grouping
import json
import sys

from story_app import StoryConfig
```

#### Type Safety Issues (mypy)
```python
# ❌ Problem: Optional list fields caused widespread type errors
@dataclass
class StoryConfig:
    themes: Optional[List[str]] = None

# ✅ Solution: Use field(default_factory=list)
from dataclasses import dataclass, field

@dataclass  
class StoryConfig:
    themes: List[str] = field(default_factory=list)
```

#### Critical Bug: Misplaced Code
```python
# ❌ Problem: Accidentally duplicated initialization outside __init__
class StoryGenerator:
    def __init__(self, config: StoryConfig):
        self._dry_run = False
        # ... other init code
    
    # This was accidentally added outside __init__ (indentation error!)
    self._dry_run = config.dry_run  # NameError: self not defined
    self._prompt_log = []

# ✅ Solution: Keep all attribute initialization inside __init__
class StoryGenerator:
    def __init__(self, config: StoryConfig):
        self._dry_run = config.dry_run
        self._prompt_log = []
        self._branch_overrides = {}
```

#### Testing Challenges
```python
# ❌ Problem: Tests failed in different environments
def test_cli():
    result = subprocess.run(["storygen", "--help"])  # May not be in PATH

# ✅ Solution: Robust executable discovery
def _resolve_storygen_exe():
    exe = shutil.which("storygen")
    if exe:
        return exe
    candidate = Path(sys.executable).with_name("storygen")
    if candidate.exists():
        return str(candidate)
    return None
```

### Phase 5: CI/CD & Automation
**Goal**: Automated quality gates and multi-Python support
**Implementation**:
- GitHub Actions workflow testing Python 3.9-3.12
- Automated linting (Ruff), type checking (mypy), and testing (pytest)
- Matrix builds to catch environment-specific issues

### 📝 Key Development Lessons

1. **Start Simple**: Begin with deterministic generation, add AI as optional enhancement
2. **Offline First**: Don't require internet/models for basic functionality  
3. **Graceful Degradation**: Provide fallbacks for optional dependencies
4. **Multiple Interfaces**: CLI wizard, programmatic CLI, and web UI serve different users
5. **Quality Gates**: Linting, type checking, and tests catch issues early
6. **Package Properly**: Use modern `pyproject.toml` with entry points and extras
7. **Document the Journey**: Show mistakes and fixes to help other developers

### 🐛 Common Pitfalls Avoided

- **Don't**: Make AI/internet required for basic functionality
- **Don't**: Use `Optional[List[str]] = None` in dataclasses (use `field(default_factory=list)`)
- **Don't**: Put attribute initialization outside `__init__` methods
- **Don't**: Assume executables are in PATH in tests
- **Don't**: Ignore import ordering and type safety from the start

## � Contributing

We welcome contributions! Here are several ways to help:

### Adding Content
- **Story Templates**: Add new story structures to `templates.json`
- **World Building**: Expand `worlds.json` with new settings and cultures
- **Example Configurations**: Create new JSON configs in `examples/`

### Code Contributions
- Follow the established patterns shown in the codebase
- Run quality checks: `ruff check . && mypy . && pytest`
- Add tests for new functionality
- Update documentation for new features

### Sharing & Feedback
- Share screenshots/GIFs of the Streamlit app in action
- Report bugs with reproduction steps
- Suggest new story elements or export formats
- Share example outputs and use cases

### Development Setup
```bash
git clone https://github.com/yourusername/story-generator.git
cd story-generator
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e .[dev]
```

## 🗺️ Roadmap

Future enhancements being considered:
- **Character Development**: Detailed character arc tracking
- **Plot Visualization**: Generate story structure diagrams
- **Collaborative Features**: Multi-user story building
- **Advanced Export**: PDF with custom formatting
- **Plugin System**: Custom story generators and themes
- **Analytics**: Story complexity and readability metrics

## 📄 License

MIT License - see LICENSE file for details.

## 🏷️ Tags

`#python` `#storytelling` `#creative-writing` `#cli-tool` `#streamlit` `#offline-first` `#educational` `#rpg-tools` `#narrative-generation` `#writing-assistant`

---

**Built with ❤️ for writers, teachers, and storytellers everywhere.**