import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional pretty console
try:
    from rich.console import Console
    from rich.table import Table
    _RICH_AVAILABLE = True
    _console = Console()
except Exception:
    _RICH_AVAILABLE = False
    _console = None

def _choose_from_list(prompt: str, options: List[str], default: Optional[str] = None) -> str:
    """Simple numbered menu using Rich when available; returns selected string.
    Falls back to input() free text with default when invalid.
    """
    try:
        if _RICH_AVAILABLE and sys.stdin and sys.stdin.isatty():
            _console.print(f"[bold]{prompt}[/]")
            for i, opt in enumerate(options, 1):
                _console.print(f"  [cyan]{i}[/]. {opt}")
            ans = input("> ").strip()
            if ans.isdigit() and 1 <= int(ans) <= len(options):
                return options[int(ans) - 1]
        # Fallback
        entered = input(f"{prompt} [{default or options[0]}]: ") or (default or options[0])
        # If user typed a number, honor it
        if entered.isdigit() and 1 <= int(entered) <= len(options):
            return options[int(entered) - 1]
        # If they typed a value that matches, use it; else default
        return entered if entered in options else (default or options[0])
    except Exception:
        return default or options[0]

@dataclass
class StoryConfig:
    genre: str = "speculative fiction"
    tone: str = "hopeful"
    pov: str = "third"
    tense: str = "past"
    length: str = "short"  # flash (300–700), short (1k–2k), novelette (7k–12k), chapter (2k–4k)
    audience: str = "adult"
    rating: str = "PG-13"
    ending: str = "bittersweet"
    themes: List[str] = field(default_factory=lambda: ["redemption"])
    required_elements: List[str] = field(default_factory=list)
    setting: str = "generic modern world"
    narrative_structure: str = "3-act"
    # New story shape and world variables
    arc_style: Optional[str] = None  # rise-fall | fall-rise | steady climb
    protagonist_age: Optional[str] = None  # teen | adult | elder
    protagonist_role: Optional[str] = None  # outsider | reluctant hero | etc.
    time_period: Optional[str] = None  # medieval | modern | futuristic | etc.
    tech_level: Optional[str] = None  # primitive | industrial | futuristic
    magic_system: Optional[str] = None  # none | ritualistic | forbidden
    weather_mood: Optional[str] = None  # stormy | fog-laden | etc.
    cultural_conflict: Optional[str] = None  # tradition vs progress, etc.
    # Optional character customization attributes (set dynamically via CLI or world archetypes)
    protagonist: Optional[str] = None
    protagonist_want: Optional[str] = None
    protagonist_need: Optional[str] = None
    protagonist_flaw: Optional[str] = None
    antagonist: Optional[str] = None
    antagonist_goal: Optional[str] = None
    ally: Optional[str] = None
    ally_trait: Optional[str] = None
    ally_secret: Optional[str] = None

    def __post_init__(self):
        # Defaults for newly added world/arc variables
        if self.arc_style is None:
            self.arc_style = "steady climb"
        if self.protagonist_age is None:
            self.protagonist_age = "adult"
        if self.protagonist_role is None:
            self.protagonist_role = ""
        if self.time_period is None:
            self.time_period = "modern"
        if self.tech_level is None:
            self.tech_level = "industrial"
        if self.magic_system is None:
            self.magic_system = "none"
        if self.weather_mood is None:
            self.weather_mood = ""
        if self.cultural_conflict is None:
            self.cultural_conflict = ""

class StoryGenerator:
    # Class-level annotations for instance attributes initialized in __init__
    _prompt_log: List[Dict[str, str]]
    _branch_overrides: Dict[str, str]
    def __init__(self, config: StoryConfig, load_model: bool = True, deterministic: bool = False, dry_run: bool = False):
        self.config = config
        self.deterministic = bool(deterministic)
        self.word_counts = {
            "flash": (300, 700),
            "short": (1000, 2000),
            "novelette": (7000, 12000),
            "chapter": (2000, 4000)
        }
        # Character defaults (may be overridden via CLI or archetypes)
        self.characters = {
            "protagonist": {
                "name": (getattr(config, 'protagonist', None) or "Alex"),
                "want": (getattr(config, 'protagonist_want', None) or "freedom"),
                "need": (getattr(config, 'protagonist_need', None) or "connection"),
                "flaw": (getattr(config, 'protagonist_flaw', None) or "distrust"),
            },
            "antagonist": {
                "name": (getattr(config, 'antagonist', None) or "Rival"),
                "goal": (getattr(config, 'antagonist_goal', None) or "control"),
            },
            "ally": {
                "name": getattr(config, 'ally', None) or "",  # empty if unspecified
                "trait": getattr(config, 'ally_trait', None) or "steadfast",
                "secret": getattr(config, 'ally_secret', None) or "unspoken debt"
            },
        }
        # validate and normalize config to avoid runtime IndexErrors
        self._validate_and_normalize_config()
        # Try loading external templates
        self.external_templates = self._load_external_templates()
        # Try loading world data to augment elements
        self.worlds = self._load_worlds()
        if self.worlds:
            # If the current setting exists in worlds, extend required_elements
            setting_key = self.config.setting
            world_data = self.worlds.get('settings', {}).get(setting_key)
            if world_data and world_data.get('elements'):
                for e in world_data.get('elements'):
                    if e not in self.config.required_elements:
                        self.config.required_elements.append(e)
            # Apply archetypes only if user did not explicitly override via CLI
            if world_data and world_data.get('archetypes'):
                # Detect overrides: user override if a non-empty value provided
                user_overrode_protagonist = bool(getattr(config, 'protagonist', None))
                user_overrode_antagonist = bool(getattr(config, 'antagonist', None))
                arche = world_data.get('archetypes', {})
                if not user_overrode_protagonist:
                    self.characters['protagonist'].update(arche.get('protagonist', {}))
                if not user_overrode_antagonist:
                    self.characters['antagonist'].update(arche.get('antagonist', {}))
                # Ally archetype (apply only if user did not provide ally)
                if not getattr(config, 'ally', None) and 'ally' in arche:
                    ally_arch = arche.get('ally', {})
                    self.characters['ally'].update(ally_arch)
                    # also set on config for prompt usage
                    self.config.ally = self.characters['ally'].get('name')
                    self.config.ally_trait = self.characters['ally'].get('trait')
                    self.config.ally_secret = self.characters['ally'].get('secret')
        # Final safety: ensure protagonist/antagonist have defaults if still missing/empty
        if not self.characters['protagonist']['name']:
            self.characters['protagonist']['name'] = 'Alex'
        if not self.characters['antagonist']['name']:
            self.characters['antagonist']['name'] = 'Rival'

        # Language model (lazy): allow but don't load until needed
        self.generator = None
        self._gen_kwargs = None
        self._allow_model = bool(load_model)
        # Dry-run mode: record prompts, skip LM calls
        self._dry_run = bool(dry_run)
        self._prompt_log = []
        # Branch override hooks (e.g., for Streamlit UI)
        self._branch_overrides = {}

    def set_branch_override(self, section: str, value: str) -> None:
        """Force a branch decision for interactive moments (e.g., 'midpoint.choice' -> 'y'|'n')."""
        if section and value:
            self._branch_overrides[section] = value

    def _seeded_choice(self, seq: List[str], section: str) -> str:
        """Stable random choice per section for reproducibility."""
        if not seq:
            return ""
        try:
            seed_val = hashlib.md5(section.encode()).hexdigest()
            rnd = random.Random(int(seed_val, 16))
            return rnd.choice(seq)
        except Exception:
            return seq[0]

    def _paraphrase_variants(self, text: str) -> List[str]:
        """Cheap paraphrase variants without external models."""
        if not text:
            return [""]
        variants = [
            text,
            text.replace(" but ", " yet "),
            text.replace(" and ", " as well as "),
            text.replace(" faces", " confronts"),
        ]
        # deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    def _load_model(self):
        """Load a small, free model with multi-model fallback; support deterministic sampling."""
        if not self._allow_model or self.generator is not None:
            return
        try:
            from transformers import pipeline
        except Exception:
            self.generator = None
            self._gen_kwargs = None
            return
        # Priority list: try TinyLlama 4-bit, then TinyLlama, then GPT-Neo 125M, DistilGPT2, GPT-2
        candidates = [
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", {"load_in_4bit": True}),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", {}),
            ("EleutherAI/gpt-neo-125M", {}),
            ("distilgpt2", {}),
            ("gpt2", {}),
        ]
        for name, kwargs in candidates:
            try:
                self.generator = pipeline(
                    "text-generation",
                    model=name,
                    device_map="auto",
                    model_kwargs=kwargs
                )
                break
            except Exception:
                self.generator = None
                continue
        if not self.generator:
            self._gen_kwargs = None
            return
        # Base generation kwargs
        self._gen_kwargs = dict(
            max_new_tokens=150,
            do_sample=(not self.deterministic),
            temperature=(0.7 if not self.deterministic else 0.0),
            top_p=(0.9 if not self.deterministic else 1.0),
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            return_full_text=False,
            truncation=True,
        )
        try:
            pad_id = getattr(self.generator.tokenizer, 'pad_token_id', None)
            if pad_id is None:
                pad_id = getattr(self.generator.tokenizer, 'eos_token_id', None)
            if pad_id is not None:
                self._gen_kwargs['pad_token_id'] = pad_id
        except Exception:
            pass

    def _ensure_model(self):
        if self.generator is None and self._allow_model:
            self._load_model()

    def _generate_with_cache(self, prompt: str, section: str, max_new_tokens: Optional[int] = None) -> Optional[str]:
        """File-based cache for LM outputs. Returns None if no model or generation fails."""
        # Attempt to read cache first
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        key_src = f"{section}|{prompt}".encode('utf-8', errors='ignore')
        cache_key = hashlib.md5(key_src).hexdigest()
        cache_path = os.path.join(cache_dir, f"{section}_{cache_key}.txt")
        # Always log the prompt
        try:
            self._prompt_log.append({"section": section, "prompt": prompt})
        except Exception:
            pass
        # In dry-run, don't call the model
        if self._dry_run:
            return None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass
        # Generate if possible
        self._ensure_model()
        if not self.generator:
            return None
        try:
            gen_kwargs = dict(self._gen_kwargs or {})
            if max_new_tokens is not None:
                gen_kwargs['max_new_tokens'] = max_new_tokens
            out = self.generator(prompt, **gen_kwargs)
            first = out[0] if isinstance(out, list) and out else out
            generated = first.get('generated_text') if isinstance(first, dict) else first
            cleaned = self._clean_model_output(generated)
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
            except Exception:
                pass
            return cleaned
        except Exception:
            return None

    def _safe_choice(self, seq: List[str], fallback: str) -> str:
        """Return a random choice from seq or a fallback if seq is empty or None."""
        if seq:
            try:
                return random.choice(seq)
            except (IndexError, TypeError):
                return fallback
        return fallback

    def _strip_leading_article(self, text: str) -> str:
        """Remove a leading 'the', 'a', or 'an' (case-insensitive) from a string."""
        if not text:
            return text
        return re.sub(r'^(?:the|a|an)\s+', '', text, flags=re.IGNORECASE)

    def _format_with_optional_article(self, element: str, add_article: bool = True) -> str:
        """Return the element optionally prefixed with an indefinite article.

        Heuristic: if element appears plural or multi-word, skip the article.
        """
        if not element:
            return element
        elem = element.strip()
        # if it already has a leading article, leave it normalized
        if re.match(r'^(?:the|a|an)\b', elem, flags=re.IGNORECASE):
            return elem
        if not add_article:
            return elem
        # heuristics: multi-word or plural (endswith s) -> skip 'a'
        if ' ' in elem or elem.endswith('s'):
            return elem
        # default: use 'a' (simple; not handling 'an' thoroughly)
        return f"a {elem}"

    def _validate_and_normalize_config(self) -> None:
        """Ensure lists are present and normalize themes to avoid duplicate 'the'."""
        # Ensure lists are lists
        if self.config.themes is None:
            self.config.themes = ["redemption"]
        if self.config.required_elements is None:
            self.config.required_elements = []
        # Normalize themes to strip leading articles when templates also include 'the'
        self.config.themes = [self._strip_leading_article(t) for t in self.config.themes]

    def _load_external_templates(self) -> Optional[Dict]:
        """Load templates; support both flat lists and setting-keyed dicts per category."""
        try:
            base = os.path.dirname(__file__)
            path = os.path.join(base, 'templates.json')
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            templates = {
                'hook': data.get('hook_templates') or data.get('hook') or [],
                'complication': data.get('complication_templates') or data.get('complication') or [],
                'filler': data.get('filler_templates') or data.get('filler') or []
            }
            return templates
        except Exception:
            return None

    def _load_worlds(self) -> Optional[Dict]:
        """Load worlds.json or worlds.json.gz; ensure defaults for factions/artifacts/culture."""
        import gzip
        try:
            base = os.path.dirname(__file__)
            gz = os.path.join(base, 'worlds.json.gz')
            path = os.path.join(base, 'worlds.json')
            data = None
            if os.path.exists(gz):
                with gzip.open(gz, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            elif os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                return {"settings": {}}
            for _, details in data.get('settings', {}).items():
                details.setdefault('factions', [])
                details.setdefault('artifacts', [])
                details.setdefault('culture', {'customs': [], 'values': [], 'conflicts': []})
            return data
        except Exception:
            return {"settings": {}}

    def _clean_model_output(self, text: str) -> str:
        """Clean language model output to remove incomplete sentences and align POV/tense heuristically."""
        if not text:
            return text
        # Truncate weird trailing tokens often produced by small models
        text = text.replace('\n', ' ').strip()
        # Remove likely prompt echo: if the prompt appears at the start, cut it out
        # Build a short prompt heuristic used in generate_hook
        try:
            prompt_prefix = f"Write a {self.config.tone} opening paragraph for a {self.config.genre} story set in {self.config.setting}."
            if text.startswith(prompt_prefix):
                # remove the prompt prefix
                text = text[len(prompt_prefix):].strip()
        except Exception:
            pass
        # Split into sentences (very simple)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Keep sentences of reasonable length
        kept = [s for s in sentences if len(s.strip()) > 10]
        if not kept:
            out = text[:500]
        else:
            out = ' '.join(kept)
        # Basic POV fix: naive pronoun swap for first-person preference
        if self.config.pov == 'first':
            out = re.sub(r"\b(they|he|she)\b", 'I', out, flags=re.IGNORECASE)
        return out[:500]

    def generate_hook(self) -> str:
        """Opening scene influenced by weather, role, and time period."""
        protagonist = self.characters['protagonist']['name']
        element = random.choice(self.config.required_elements or ["mysterious object"])
        role = ""
        if getattr(self.config, 'protagonist_role', None):
            pr = str(self.config.protagonist_role).strip()
            art = 'an' if re.match(r'^[aeiou]', pr, flags=re.IGNORECASE) else 'a'
            role = f"as {art} {pr}"
        mood = f" Beneath {self.config.weather_mood} skies," if getattr(self.config, 'weather_mood', None) else ""
        time_period = self.config.time_period or "modern"
        base = (
            f"{mood} In the {time_period} {self.config.setting}, "
            f"{protagonist} began their path {role}, drawn to {element} and the theme of {self.config.themes[0]}."
        )
        return self._seeded_choice(self._paraphrase_variants(base), section="hook")

    def generate_inciting_incident(self) -> str:
        """Generate an event that forces a choice or reveals stakes."""
        action = random.choice(["discovers", "receives", "stumbles upon"])
        element = self._safe_choice(self.config.required_elements, "mysterious object")
        element_phrase = self._format_with_optional_article(element, add_article=True)
        return (
            f"When {self.characters['protagonist']['name']} {action} {element_phrase}, "
            f"the path to their {self.characters['protagonist']['want']} opens—but at a cost they never expected."
        )

    def generate_complications(self) -> List[str]:
        """Escalating problems, shaped by arc style and cultural conflict."""
        protagonist = self.characters['protagonist']['name']
        antagonist = self.characters['antagonist']['name']
        flaws = self.characters['protagonist']['flaw']
        tech = self.config.tech_level
        conflict = f" Cultural rifts emerge: {self.config.cultural_conflict}." if getattr(self.config, 'cultural_conflict', None) else ""
        arc = {
            "rise-fall": f"But triumph fades: {antagonist} twists victory into setback.",
            "fall-rise": f"Defeat seems certain, yet {protagonist} glimpses a fragile hope.",
            "steady climb": f"Each obstacle grows harder, testing {protagonist}'s endurance."
        }.get(self.config.arc_style or "", "")
        items = [
            f"{protagonist}'s {flaws} drives them into error, deepening tension.",
            f"{antagonist} exploits {tech} advantage to block {protagonist}.{conflict}",
            arc
        ]
        # Expanded world details
        world = (self.worlds or {}).get('settings', {}).get(self.config.setting, {}) if self.worlds else {}
        climate = (world or {}).get('climate', '')
        currency = (world or {}).get('currency', '')
        arch = (world or {}).get('architecture', '')
        myth = self._safe_choice((world or {}).get('myths', []), "")
        extra_world: List[str] = []
        if climate:
            extra_world.append(f"The climate is {climate}.")
        if currency:
            extra_world.append(f"Markets bustle with {currency}.")
        if arch:
            extra_world.append(f"Buildings of {arch} line the streets.")
        if myth:
            extra_world.append(f"A local myth whispers of {myth}.")
        if extra_world:
            items[-1] = (items[-1] + " " + " ".join(extra_world)).strip()
        # Paraphrase with stable selection per line
        out: List[str] = []
        for idx, line in enumerate(items):
            variants = self._paraphrase_variants(line)
            out.append(self._seeded_choice(variants, section=f"complications:{idx}"))
        return out

    def generate_midpoint(self) -> str:
        """Midpoint with a simple branching choice; deterministic fallback when non-interactive."""
        protagonist = self.characters['protagonist']['name']
        element = self._safe_choice(self.config.required_elements, "mysterious object")
        base_text = f"At midpoint, {protagonist} discovers {element}, shifting everything."
        # Attempt interactive branch; fallback to deterministic choice when not a TTY or in dry-run
        # Check override first
        choice_val = (self._branch_overrides.get('midpoint.choice') or None)
        try:
            if choice_val not in ['y', 'n'] and not self._dry_run and sys.stdin and sys.stdin.isatty():
                ans = input(f"Should {protagonist} trust the {element}? (y/n): ").strip().lower()
                choice_val = 'y' if ans.startswith('y') else 'n'
            else:
                if choice_val not in ['y', 'n']:
                    choice_val = self._seeded_choice(['y','n'], section='midpoint.choice')
        except Exception:
            choice_val = self._seeded_choice(['y','n'], section='midpoint.choice')
        tail = " Trust opens a fragile path forward." if choice_val == 'y' else " Distrust twists the path into danger."
        text = base_text + tail
        return self._seeded_choice(self._paraphrase_variants(text), section="midpoint")

    def generate_crisis(self) -> str:
        """Worst moment, infused with weather + arc style."""
        protagonist = self.characters['protagonist']['name']
        mood = f"As {self.config.weather_mood} skies bear down, " if getattr(self.config, 'weather_mood', None) else ""
        arc = {
            "rise-fall": "The false victory shatters completely.",
            "fall-rise": "The bleakest defeat hides a spark of salvation.",
            "steady climb": "The next step threatens collapse under the weight."
        }.get(self.config.arc_style or "", "")
        base = (
            f"{mood}{protagonist} confronts their flaw ({self.characters['protagonist']['flaw']}), "
            f"and must choose their true need ({self.characters['protagonist']['need']}). {arc}"
        )
        return self._seeded_choice(self._paraphrase_variants(base), section="crisis")

    def generate_climax(self) -> str:
        """Final confrontation shaped by structure + cultural conflict."""
        protagonist = self.characters['protagonist']['name']
        antagonist = self.characters['antagonist']['name']
        structure = {
            "3-act": "The final act converges into direct confrontation.",
            "hero’s journey": "The return home looms, yet the final trial awaits.",
            "episodic": "Another chapter closes, but this confrontation defines the arc.",
            "vignette": "The moment crystallizes in fleeting intensity."
        }.get(self.config.narrative_structure, "The confrontation arrives.")
        conflict = f" Their struggle embodies {self.config.cultural_conflict}." if getattr(self.config, 'cultural_conflict', None) else ""
        base = f"{structure} {protagonist} faces {antagonist} in the {self.config.setting}.{conflict}"
        return self._seeded_choice(self._paraphrase_variants(base), section="climax")

    def generate_resolution(self) -> str:
        """Generate the emotional payoff with arc-aware closure (LM-enhanced if available)."""
        protagonist = self.characters['protagonist']['name']
        element = self._safe_choice(self.config.required_elements, 'mysterious object')
        world = self.worlds.get('settings', {}).get(self.config.setting, {}) if self.worlds else {}
        setting_desc = world.get('description', f"the {self.config.setting}")
        setting_desc = re.sub(r'^(?:[Tt]he\s+)+', 'the ', setting_desc).strip()
        ally = self.characters.get('ally', {}).get('name') or None
        ally_line = f" Mention how the ally {ally} adjusts in the aftermath." if ally else ""
        if self.generator and not self._dry_run:
            rating_instruction = "Keep imagery appropriate for a PG-13 rating; avoid graphic or explicit content." if self.config.rating.upper() == 'PG-13' else "Maintain content appropriate for the declared rating."
            ending_instruction = {
                'bittersweet': 'Convey mixed emotions: a sacrifice or cost, yet a glimmer of growth.',
                'triumphant': 'Convey clear victory and emotional uplift.',
                'open-ended': 'Convey ambiguity; end on an unresolved but evocative note.'
            }.get(self.config.ending, 'Convey a reflective emotional close.')
            genre_prompts = {
                "cozy mystery": "Use a light, engaging tone with subtle clues and cozy imagery.",
                "low fantasy noir": "Emphasize gritty atmosphere, moral ambiguity, and sparse dialogue.",
                "science fiction": "Highlight speculative technology with grounded human stakes.",
                "sci-fantasy": "Blend wonder with tactile, grounded details; keep the magic rule-based.",
                "romance": "Lean into warm, character-driven beats and emotional subtext.",
            }
            genre_instruction = genre_prompts.get(self.config.genre, "Use vivid, immersive imagery.")
            prompt = (
                f"Write a closing paragraph for a {self.config.tone} {self.config.genre} story set in {setting_desc}. "
                f"{genre_instruction} {ending_instruction} {rating_instruction} The protagonist, {protagonist}, reflects on the theme of {self.config.themes[0]} "
                f"with reference to {self._format_with_optional_article(element)}.{ally_line} Use {self.config.pov} person and {self.config.tense} tense. Avoid repetition."
            )
            try:
                out = self.generator(prompt, **(self._gen_kwargs or {'max_length': 120, 'num_return_sequences': 1}))
                first = out[0] if isinstance(out, list) and out else out
                generated = first.get('generated_text') if isinstance(first, dict) else first
                return self._clean_model_output(generated)
            except Exception:
                pass
        # Arc-aware closure fallback
        arc = self.config.arc_style or ""
        closure_map = {
            "rise-fall": f"Though {protagonist} fought bravely, irony remains: victory tastes like loss.",
            "fall-rise": f"Against all odds, {protagonist} finds fragile hope, carrying scars into tomorrow.",
            "steady climb": f"{protagonist} endures, the path still long but their spirit steadier than before.",
        }
        if arc in closure_map:
            return closure_map[arc]
        # fallback to previous ending-style mapping if arc unknown
        ending_types = {
            "bittersweet": (
                f"{protagonist} walks away, the {self._strip_leading_article(self._safe_choice(self.config.required_elements, 'mysterious object'))} left behind, "
                f"a quiet {self.config.themes[0]} settling in their heart." + (f" Beside them, {ally} keeps an unresolved promise." if ally else "")
            ),
            "triumphant": (
                f"With the {self._strip_leading_article(self._safe_choice(self.config.required_elements, 'mysterious object'))} in hand, "
                f"{protagonist} stands taller, their {self.config.themes[0]} won." + (f" {ally} laughs, already planning what comes next." if ally else "")
            ),
            "open-ended": (
                f"The {self._strip_leading_article(self._safe_choice(self.config.required_elements, 'mysterious object'))} remains, its meaning unclear, "
                f"as {protagonist} steps into the unknown." + (f" Somewhere, {ally} vanishes into the mist." if ally else "")
            )
        }
        return ending_types.get(self.config.ending, ending_types["bittersweet"])

    def generate_story(self) -> str:
        """Generate a complete story based on the config."""
        word_target = random.randint(*self.word_counts[self.config.length])
        story_parts = [
            self.generate_hook(),
            self.generate_inciting_incident(),
            "\n".join(self.generate_complications()),
            self.generate_midpoint(),
            self.generate_crisis(),
            self.generate_climax(),
            self.generate_resolution()
        ]
        story = "\n\n".join(story_parts)

        # Build varied micro-sentences to expand the story naturally until we reach word target
        if isinstance(self.external_templates, dict) and self.external_templates.get('filler'):
            # Convert string templates into callables that format the placeholders
            def make_callable(tmpl: str):
                def call():
                    # choose a random element for {element}
                    element = self._safe_choice(self.config.required_elements, '')
                    return tmpl.format(
                        setting=self.config.setting,
                        theme=self.config.themes[0],
                        protagonist=self.characters['protagonist']['name'],
                        element=element
                    )
                return call
            micro_templates = [make_callable(t) for t in self.external_templates.get('filler', [])]
        else:
            micro_templates = [
                lambda: f"The {self.config.setting} pulsed with {self.config.themes[0]}, urging {self.characters['protagonist']['name']} forward.",
                lambda: f"Night after night the {self.config.setting} hummed with {self.config.themes[0]}.",
                lambda: f"A memory surfaced, tied to {self.config.themes[0]}, and {self.characters['protagonist']['name']} felt it like a pull.",
                lambda: f"Shadows moved through the {self.config.setting}, carrying whispers of {self.config.themes[0]}.",
                lambda: f"Somewhere, a {self._safe_choice(self.config.required_elements, 'small sound')} echoed—a reminder of {self.config.themes[0]}.",
                lambda: f"A single streetlamp threw its halo over the {self.config.setting}, and the air tasted of {self.config.themes[0]}.",
                lambda: f"{self.characters['protagonist']['name']} touched the {self._safe_choice(self.config.required_elements, 'object')} and a cold memory returned.",
                lambda: f"Rain traced the edges of the {self.config.setting} while {self.config.themes[0]} threaded through the alleys.",
                lambda: f"An old song hummed from somewhere—half a nursery rhyme, half a warning about {self.config.themes[0]}.",
                lambda: f"The {self._safe_choice(self.config.required_elements, 'lantern')} threw small, honest light; it did not hide {self.config.themes[0]}.",
                lambda: f"{self.characters['protagonist']['name']} recalled a face from the {self.config.setting}, a face shaped by {self.config.themes[0]}.",
                lambda: f"Sounds bent oddly in the {self.config.setting}, as if the buildings remembered {self.config.themes[0]}.",
                lambda: f"A passing stranger muttered something about {self.config.themes[0]}, then vanished into the fog.",
                lambda: f"The taste of salt and smoke lingered—small proof that the {self.config.setting} kept its secrets about {self.config.themes[0]}.",
                lambda: f"{self.characters['protagonist']['name']} thought of what they'd give up for {self.characters['protagonist']['want']}, and the {self.config.themes[0]} that came with it.",
                lambda: f"Footsteps echoed, uneven and urgent, and every step seemed to measure the cost of {self.config.themes[0]}.",
                lambda: f"The {self._safe_choice(self.config.required_elements, 'lockbox')} sat like a quiet promise, heavy with {self.config.themes[0]}.",
                lambda: f"An old photograph, creased and warm from being held, suggested a different ending—one that denied {self.config.themes[0]}.",
                lambda: "A brief laugh broke the night's rhythm; it sounded like memory trying to surface.",
                lambda: f"The air smelled of rust and rain; even the weather seemed to argue about {self.config.themes[0]}.",
                lambda: f"{self.characters['ally']['name']} lingered nearby, their secret pressing against the quiet." if self.characters['ally']['name'] else "",
                lambda: f"A gesture from {self.characters['ally']['name']} anchored the moment." if self.characters['ally']['name'] else "",
            ]
            # Weather-driven filler mood
            if getattr(self.config, 'weather_mood', None):
                wm = self.config.weather_mood
                micro_templates.extend([
                    lambda wm=wm: f"The {wm} sky mirrored {self.characters['protagonist']['name']}'s turmoil.",
                    lambda: f"Storm winds carried whispers of {self.config.themes[0]}.",
                    lambda: f"Clouds pressed low over the {self.config.setting}, heavy with fate.",
                ])

        # Use shuffled paragraph assembly to reduce repeating the exact same sentence
        filler_paragraphs: List[str] = []
        recent_sentences: List[str] = []
        while len((story + ' ').split()) < word_target:
            # pick 2-4 unique micro-templates and join them into a paragraph
            choices = random.sample(micro_templates, k=random.randint(2, min(4, len(micro_templates))))
            generated_lines = []
            for c in choices:
                line = c()
                if not line:
                    continue
                # de-dup against last 10 lines
                if line in recent_sentences:
                    continue
                generated_lines.append(line)
                recent_sentences.append(line)
                if len(recent_sentences) > 10:
                    recent_sentences.pop(0)
            paragraph = ' '.join(generated_lines)
            if not paragraph:
                # fallback to a simple non-empty sentence
                paragraph = f"The {self.config.setting} breathed with {self.config.themes[0]}."
            filler_paragraphs.append(paragraph)
            story += "\n\n" + paragraph

        # Trim to exact word target (safe and deterministic for length)
        words = story.split()
        trimmed = " ".join(words[:word_target])
        return trimmed

    def generate_outline(self) -> Dict[str, str]:
        """Generate a high-level outline."""
        return {
            "Hook": self.generate_hook(),
            "Inciting Incident": self.generate_inciting_incident(),
            "Complications": "\n".join(self.generate_complications()),
            "Midpoint": self.generate_midpoint(),
            "Crisis": self.generate_crisis(),
            "Climax": self.generate_climax(),
            "Resolution": self.generate_resolution()
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a short story outline and draft.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    parser.add_argument("--length", choices=["flash", "short", "novelette", "chapter"], default="short")
    parser.add_argument("--themes", type=str, help="Comma-separated themes (e.g. 'memory,loss')")
    parser.add_argument("--elements", type=str, help="Comma-separated required elements (e.g. 'lockbox,stray dog')")
    parser.add_argument("--pov", choices=["first", "third"], help="Point of view")
    parser.add_argument("--tense", choices=["present", "past"], help="Tense")
    parser.add_argument("--ending", choices=["bittersweet", "triumphant", "open-ended"], help="Ending type")
    parser.add_argument("--setting", type=str, help="Setting description")
    parser.add_argument("--genre", type=str, help="Genre")
    parser.add_argument("--tone", type=str, help="Tone")
    parser.add_argument("--output", choices=["story", "outline", "both"], default="both", help="Output format (story, outline, or both)")
    # Default to offline; advanced LM integration is optional
    parser.add_argument("--use-model", action="store_true", help="Enable language model generation (may download small models)")
    parser.add_argument("--no-model", action="store_true", help="Force-disable language model (overrides --use-model)")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic generation (temperature=0, top_p=1)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call model; print prompts that would be used")
    parser.add_argument("--step-mode", action="store_true", help="Reveal sections one-by-one, press Enter to continue")
    parser.add_argument("--multi-endings", type=int, default=1, help="Generate multiple alternate endings (1-3)")
    parser.add_argument("--save", type=str, help="Save output to a file (e.g., output.txt)")
    parser.add_argument("--protagonist", type=str, help="Protagonist name")
    parser.add_argument("--protagonist-want", dest="protagonist_want", type=str, help="Protagonist's want")
    parser.add_argument("--protagonist-need", dest="protagonist_need", type=str, help="Protagonist's need")
    parser.add_argument("--protagonist-flaw", dest="protagonist_flaw", type=str, help="Protagonist's flaw")
    parser.add_argument("--antagonist", type=str, help="Antagonist name")
    parser.add_argument("--antagonist-goal", dest="antagonist_goal", type=str, help="Antagonist's goal")
    parser.add_argument("--ally", type=str, help="Ally character name")
    parser.add_argument("--ally-trait", dest="ally_trait", type=str, help="Ally's notable trait")
    parser.add_argument("--ally-secret", dest="ally_secret", type=str, help="Ally's secret or hidden agenda")
    parser.add_argument("--preview-templates", action="store_true", help="Preview available templates and exit")
    parser.add_argument("--list-settings", action="store_true", help="List available settings from worlds.json and exit")
    parser.add_argument("--save-config", type=str, help="Save configuration to a JSON file")
    parser.add_argument("--load-config", type=str, help="Load configuration from a JSON file")
    parser.add_argument("--wizard", action="store_true", help="Use interactive configuration wizard")
    parser.add_argument("--add-setting", type=str, help="Add a new setting to worlds.json (name)")
    parser.add_argument("--yes", action="store_true", help="Skip configuration confirmation prompt")
    parser.add_argument("--save-profile", type=str, help="Save final configuration to a named profile (~/.storygen_profiles/<name>.json)")
    parser.add_argument("--load-profile", type=str, help="Load configuration from a named profile (~/.storygen_profiles/<name>.json)")
    parser.add_argument("--cache-story", action="store_true", help="Cache full story/outline by config hash for instant replays")
    parser.add_argument("--world-pack", type=str, help="Merge additional worlds_<pack>.json before generation")
    # New world/arc variables
    parser.add_argument("--arc-style", dest="arc_style", type=str, choices=["rise-fall", "fall-rise", "steady climb"], help="Arc style shaping complications/crisis")
    parser.add_argument("--protagonist-age", dest="protagonist_age", type=str, choices=["teen", "adult", "elder"], help="Protagonist age affecting tone")
    parser.add_argument("--protagonist-role", dest="protagonist_role", type=str, help="Role like outsider, reluctant hero, etc.")
    parser.add_argument("--time-period", dest="time_period", type=str, help="Time period like medieval, modern, futuristic")
    parser.add_argument("--tech-level", dest="tech_level", type=str, choices=["primitive", "industrial", "futuristic"], help="Technology level for threats")
    parser.add_argument("--magic-system", dest="magic_system", type=str, help="Magic system name or 'none'")
    parser.add_argument("--weather-mood", dest="weather_mood", type=str, help="Weather mood words like stormy, fog-laden")
    parser.add_argument("--cultural-conflict", dest="cultural_conflict", type=str, help="High-level societal tension like 'tradition vs progress'")
    args = parser.parse_args()

    # Load configuration if requested (pre-populate args before interactive prompts)
    # Named profile support
    if getattr(args, 'load_profile', None):
        profile_dir = Path.home() / ".storygen_profiles"
        ppath = profile_dir / f"{args.load_profile}.json"
        try:
            if ppath.exists():
                with open(ppath, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                for k, v in cfg.items():
                    if getattr(args, k, None) in [None, '', False]:
                        setattr(args, k, v)
        except Exception as e:
            print(f"Failed to load profile {ppath}: {e}")
    if getattr(args, 'load_config', None):
        try:
            with open(args.load_config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                # Only set if not already provided via CLI
                if getattr(args, k, None) in [None, '', False]:
                    setattr(args, k, v)
        except Exception as e:
            print(f"Failed to load config from {args.load_config}: {e}")

    # Interactive prompts with validation
    try:
        if not args.genre:
            args.genre = _choose_from_list(
                "Choose a genre:",
                ["cozy mystery", "low fantasy noir", "science fiction", "sci-fantasy", "speculative fiction"],
                default="speculative fiction",
            )
        args.tone = args.tone or input("Enter tone (e.g., gritty) [default: hopeful]: ") or "hopeful"
        args.setting = args.setting or input("Enter setting (e.g., rain-drenched city, 1920s) [default: generic modern world]: ") or "generic modern world"
        args.pov = args.pov or input("Enter POV (first, third) [default: third]: ") or "third"
        while args.pov not in ["first", "third"]:
            print("Invalid POV. Choose 'first' or 'third'.")
            args.pov = input("Enter POV (first, third) [default: third]: ") or "third"
        args.tense = args.tense or input("Enter tense (present, past) [default: past]: ") or "past"
        while args.tense not in ["present", "past"]:
            print("Invalid tense. Choose 'present' or 'past'.")
            args.tense = input("Enter tense (present, past) [default: past]: ") or "past"
        args.ending = args.ending or input("Enter ending (bittersweet, triumphant, open-ended) [default: bittersweet]: ") or "bittersweet"
        while args.ending not in ["bittersweet", "triumphant", "open-ended"]:
            print("Invalid ending. Choose 'bittersweet', 'triumphant', or 'open-ended'.")
            args.ending = input("Enter ending (bittersweet, triumphant, open-ended) [default: bittersweet]: ") or "bittersweet"
        themes_input = args.themes or input("Enter themes (comma-separated, e.g., memory,loss) [default: redemption]: ") or "redemption"
        elements_input = args.elements or input("Enter required elements (comma-separated, e.g., lockbox,stray dog) [default: none]: ") or ""
        # New variables (optional prompts)
        args.arc_style = args.arc_style or ((input("Arc style (rise-fall, fall-rise, steady climb) [default: steady climb]: ") or "steady climb").strip().lower())
        while args.arc_style not in ["rise-fall", "fall-rise", "steady climb"]:
            print("Invalid arc style. Choose 'rise-fall', 'fall-rise', or 'steady climb'.")
            args.arc_style = (input("Arc style (rise-fall, fall-rise, steady climb) [default: steady climb]: ") or "steady climb").strip().lower()
        args.protagonist_age = args.protagonist_age or ((input("Protagonist age (teen, adult, elder) [default: adult]: ") or "adult").strip().lower())
        while args.protagonist_age not in ["teen", "adult", "elder"]:
            print("Invalid age. Choose 'teen', 'adult', or 'elder'.")
            args.protagonist_age = (input("Protagonist age (teen, adult, elder) [default: adult]: ") or "adult").strip().lower()
        args.protagonist_role = args.protagonist_role or (input("Protagonist role (e.g., outsider, reluctant hero) [default: none]: ") or "")
        args.time_period = args.time_period or (input("Time period (e.g., medieval, modern, futuristic) [default: modern]: ") or "modern")
        args.tech_level = args.tech_level or ((input("Tech level (primitive, industrial, futuristic) [default: industrial]: ") or "industrial").strip().lower())
        while args.tech_level not in ["primitive", "industrial", "futuristic"]:
            print("Invalid tech level. Choose 'primitive', 'industrial', or 'futuristic'.")
            args.tech_level = (input("Tech level (primitive, industrial, futuristic) [default: industrial]: ") or "industrial").strip().lower()
        args.magic_system = args.magic_system or (input("Magic system (e.g., none, ritualistic, forbidden) [default: none]: ") or "none")
        args.weather_mood = args.weather_mood or (input("Weather mood (e.g., stormy, fog-laden) [default: none]: ") or "")
        args.cultural_conflict = args.cultural_conflict or (input("Cultural conflict (e.g., tradition vs progress) [default: none]: ") or "")
    except Exception:
        themes_input = args.themes or "redemption"
        elements_input = args.elements or ""

    args.themes = [t.strip() for t in themes_input.split(",") if t.strip()]
    args.elements = [e.strip() for e in elements_input.split(",") if e.strip()]
    # Save configuration if requested (after parsing & interactive prompts)
    if getattr(args, 'save_config', None):
        cfg_out = {
            'seed': args.seed,
            'length': args.length,
            'themes': ','.join(args.themes),
            'elements': ','.join(args.elements),
            'pov': args.pov,
            'tense': args.tense,
            'ending': args.ending,
            'setting': args.setting,
            'genre': args.genre,
            'tone': args.tone,
            'output': args.output,
            'no_model': args.no_model,
            'deterministic': args.deterministic,
            'dry_run': args.dry_run,
            'arc_style': getattr(args, 'arc_style', None),
            'protagonist_age': getattr(args, 'protagonist_age', None),
            'protagonist_role': getattr(args, 'protagonist_role', None),
            'time_period': getattr(args, 'time_period', None),
            'tech_level': getattr(args, 'tech_level', None),
            'magic_system': getattr(args, 'magic_system', None),
            'weather_mood': getattr(args, 'weather_mood', None),
            'cultural_conflict': getattr(args, 'cultural_conflict', None),
            'protagonist': getattr(args, 'protagonist', None),
            'protagonist_want': getattr(args, 'protagonist_want', None),
            'protagonist_need': getattr(args, 'protagonist_need', None),
            'protagonist_flaw': getattr(args, 'protagonist_flaw', None),
            'antagonist': getattr(args, 'antagonist', None),
            'antagonist_goal': getattr(args, 'antagonist_goal', None),
            'ally': getattr(args, 'ally', None),
            'ally_trait': getattr(args, 'ally_trait', None),
            'ally_secret': getattr(args, 'ally_secret', None)
        }
        try:
            with open(args.save_config, 'w', encoding='utf-8') as f:
                json.dump(cfg_out, f, indent=2)
            print(f"Configuration saved to {args.save_config}")
        except Exception as e:
            print(f"Failed to save configuration to {args.save_config}: {e}")

    # Also save named profile if requested (after prompts)
    if getattr(args, 'save_profile', None):
        profile_dir = Path.home() / ".storygen_profiles"
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            ppath = profile_dir / f"{args.save_profile}.json"
            with open(ppath, 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, indent=2)
            print(f"Profile saved to {ppath}")
        except Exception as e:
            print(f"Failed to save profile: {e}")

    return args

def main():
    args = parse_args()

    # Add setting flow
    if getattr(args, 'add_setting', None):
        base = os.path.dirname(__file__)
        wpath = os.path.join(base, 'worlds.json')
        try:
            if os.path.exists(wpath):
                with open(wpath, 'r', encoding='utf-8') as f:
                    worlds = json.load(f)
            else:
                worlds = {"settings": {}}
            worlds.setdefault('settings', {})
            name = args.add_setting
            print(f"Adding setting: {name}")
            desc = input("Enter setting description: ")
            rules = [r.strip() for r in (input("Enter rules (comma-separated): ") or '').split(',') if r.strip()]
            elements = [e.strip() for e in (input("Enter elements (comma-separated): ") or '').split(',') if e.strip()]
            worlds['settings'][name] = {
                "description": desc,
                "rules": rules,
                "elements": elements,
                "archetypes": {},
                "factions": [],
                "artifacts": [],
                "culture": {"customs": [], "values": [], "conflicts": []}
            }
            with open(wpath, 'w', encoding='utf-8') as f:
                json.dump(worlds, f, indent=2)
            print(f"Added setting '{name}' to worlds.json")
        except Exception as e:
            print(f"Failed to add setting: {e}")
        return

    # If user only wants to preview templates, do so and exit early
    if getattr(args, 'preview_templates', False):
        temp_gen = StoryGenerator(StoryConfig(), load_model=False)
        if isinstance(temp_gen.external_templates, dict):
            print("Available Templates:")
            for category, templates in temp_gen.external_templates.items():
                print(f"\n{category.capitalize()}:")
                for i, t in enumerate(templates, 1):
                    print(f"  {i}. {t}")
        else:
            print("No external templates found. Create a templates.json file.")
        return

    if getattr(args, 'list_settings', False):
        tmp = StoryGenerator(StoryConfig(), load_model=False)
        if tmp.worlds and 'settings' in tmp.worlds:
            print("Available Settings:")
            for name, data in tmp.worlds['settings'].items():
                print(f"- {name}: {data.get('description', 'No description.')}")
        else:
            print("No settings found. Create a worlds.json file.")
        return

    if args.seed is not None:
        random.seed(args.seed)

    # Wizard mode builds a config interactively
    def run_wizard() -> StoryConfig:
        try:
            from prompt_toolkit import PromptSession  # noqa: E402, I001
            from prompt_toolkit.validation import Validator, ValidationError  # noqa: E402, I001
            class ChoiceValidator(Validator):
                def __init__(self, choices):
                    self.choices = choices
                def validate(self, document):
                    txt = document.text
                    if txt and txt not in self.choices:
                        raise ValidationError(message=f"Choose from {self.choices}")
            session = PromptSession()
            cfg = StoryConfig()
            cfg.genre = session.prompt("Genre [speculative fiction]: ", default="speculative fiction") or "speculative fiction"
            cfg.tone = session.prompt("Tone [hopeful]: ", default="hopeful") or "hopeful"
            cfg.setting = session.prompt("Setting [generic modern world]: ", default="generic modern world") or "generic modern world"
            cfg.pov = session.prompt("POV (first, third) [third]: ", default="third", validator=ChoiceValidator(["first","third"])) or "third"
            cfg.tense = session.prompt("Tense (present, past) [past]: ", default="past", validator=ChoiceValidator(["present","past"])) or "past"
            cfg.ending = session.prompt("Ending (bittersweet, triumphant, open-ended) [bittersweet]: ", default="bittersweet", validator=ChoiceValidator(["bittersweet","triumphant","open-ended"])) or "bittersweet"
            th = session.prompt("Themes (comma) [redemption]: ", default="redemption") or "redemption"
            el = session.prompt("Elements (comma) []: ", default="") or ""
            cfg.themes = [t.strip() for t in th.split(',') if t.strip()]
            cfg.required_elements = [e.strip() for e in el.split(',') if e.strip()]
            cfg.protagonist = session.prompt("Protagonist [Alex]: ", default="Alex") or "Alex"
            cfg.protagonist_want = session.prompt("Protagonist want [freedom]: ", default="freedom") or "freedom"
            cfg.protagonist_need = session.prompt("Protagonist need [connection]: ", default="connection") or "connection"
            cfg.protagonist_flaw = session.prompt("Protagonist flaw [distrust]: ", default="distrust") or "distrust"
            cfg.antagonist = session.prompt("Antagonist [Rival]: ", default="Rival") or "Rival"
            cfg.antagonist_goal = session.prompt("Antagonist goal [control]: ", default="control") or "control"
            cfg.ally = session.prompt("Ally (optional): ", default="") or ""
            if cfg.ally:
                cfg.ally_trait = session.prompt("Ally trait [steadfast]: ", default="steadfast") or "steadfast"
                cfg.ally_secret = session.prompt("Ally secret [unspoken debt]: ", default="unspoken debt") or "unspoken debt"
            # New variables
            cfg.arc_style = session.prompt("Arc style (rise-fall, fall-rise, steady climb) [steady climb]: ", default="steady climb", validator=ChoiceValidator(["rise-fall","fall-rise","steady climb"])) or "steady climb"
            cfg.protagonist_age = session.prompt("Protagonist age (teen, adult, elder) [adult]: ", default="adult", validator=ChoiceValidator(["teen","adult","elder"])) or "adult"
            cfg.protagonist_role = session.prompt("Protagonist role (e.g., outsider) []: ", default="") or ""
            cfg.time_period = session.prompt("Time period (e.g., medieval, modern, futuristic) [modern]: ", default="modern") or "modern"
            cfg.tech_level = session.prompt("Tech level (primitive, industrial, futuristic) [industrial]: ", default="industrial", validator=ChoiceValidator(["primitive","industrial","futuristic"])) or "industrial"
            cfg.magic_system = session.prompt("Magic system (e.g., none, ritualistic, forbidden) [none]: ", default="none") or "none"
            cfg.weather_mood = session.prompt("Weather mood (e.g., stormy) []: ", default="") or ""
            cfg.cultural_conflict = session.prompt("Cultural conflict (e.g., tradition vs progress) []: ", default="") or ""
            return cfg
        except Exception:
            # Fallback to simple input
            cfg = StoryConfig()
            cfg.genre = input("Genre [speculative fiction]: ") or "speculative fiction"
            cfg.tone = input("Tone [hopeful]: ") or "hopeful"
            cfg.setting = input("Setting [generic modern world]: ") or "generic modern world"
            cfg.pov = input("POV (first, third) [third]: ") or "third"
            cfg.tense = input("Tense (present, past) [past]: ") or "past"
            cfg.ending = input("Ending (bittersweet, triumphant, open-ended) [bittersweet]: ") or "bittersweet"
            th = input("Themes (comma) [redemption]: ") or "redemption"
            el = input("Elements (comma) []: ") or ""
            cfg.themes = [t.strip() for t in th.split(',') if t.strip()]
            cfg.required_elements = [e.strip() for e in el.split(',') if e.strip()]
            cfg.protagonist = input("Protagonist [Alex]: ") or "Alex"
            cfg.protagonist_want = input("Protagonist want [freedom]: ") or "freedom"
            cfg.protagonist_need = input("Protagonist need [connection]: ") or "connection"
            cfg.protagonist_flaw = input("Protagonist flaw [distrust]: ") or "distrust"
            cfg.antagonist = input("Antagonist [Rival]: ") or "Rival"
            cfg.antagonist_goal = input("Antagonist goal [control]: ") or "control"
            cfg.ally = input("Ally (optional): ") or ""
            if cfg.ally:
                cfg.ally_trait = input("Ally trait [steadfast]: ") or "steadfast"
                cfg.ally_secret = input("Ally secret [unspoken debt]: ") or "unspoken debt"
            # New variables
            cfg.arc_style = input("Arc style (rise-fall, fall-rise, steady climb) [steady climb]: ") or "steady climb"
            while cfg.arc_style not in ["rise-fall", "fall-rise", "steady climb"]:
                print("Invalid arc style.")
                cfg.arc_style = input("Arc style (rise-fall, fall-rise, steady climb) [steady climb]: ") or "steady climb"
            cfg.protagonist_age = input("Protagonist age (teen, adult, elder) [adult]: ") or "adult"
            while cfg.protagonist_age not in ["teen", "adult", "elder"]:
                print("Invalid age.")
                cfg.protagonist_age = input("Protagonist age (teen, adult, elder) [adult]: ") or "adult"
            cfg.protagonist_role = input("Protagonist role (e.g., outsider) []: ") or ""
            cfg.time_period = input("Time period (e.g., medieval, modern, futuristic) [modern]: ") or "modern"
            cfg.tech_level = input("Tech level (primitive, industrial, futuristic) [industrial]: ") or "industrial"
            while cfg.tech_level not in ["primitive", "industrial", "futuristic"]:
                print("Invalid tech level.")
                cfg.tech_level = input("Tech level (primitive, industrial, futuristic) [industrial]: ") or "industrial"
            cfg.magic_system = input("Magic system (e.g., none, ritualistic, forbidden) [none]: ") or "none"
            cfg.weather_mood = input("Weather mood (e.g., stormy) []: ") or ""
            cfg.cultural_conflict = input("Cultural conflict (e.g., tradition vs progress) []: ") or ""
            return cfg

    if getattr(args, 'wizard', False):
        config = run_wizard()
    else:
        config = StoryConfig(
            genre=args.genre,
            tone=args.tone,
            pov=args.pov,
            tense=args.tense,
            length=args.length,
            audience="adult",
            rating="PG-13",
            ending=args.ending,
            themes=args.themes,
            required_elements=args.elements,
            setting=args.setting,
            arc_style=args.arc_style,
            protagonist_age=args.protagonist_age,
            protagonist_role=args.protagonist_role,
            time_period=args.time_period,
            tech_level=args.tech_level,
            magic_system=args.magic_system,
            weather_mood=args.weather_mood,
            cultural_conflict=args.cultural_conflict
        )

    # Attach character customizations to config (dynamic attributes)
    for attr in [
        'protagonist', 'protagonist_want', 'protagonist_need', 'protagonist_flaw',
        'antagonist', 'antagonist_goal', 'ally', 'ally_trait', 'ally_secret'
    ]:
        if hasattr(args, attr) and getattr(args, attr) is not None:
            setattr(config, attr, getattr(args, attr))

    # Quick confirmation preview
    def confirm_config(cfg: StoryConfig) -> bool:
        if _RICH_AVAILABLE:
            _console.print("\n[bold cyan]Configuration Preview[/]:")
            table = Table(show_header=False, box=None)
            table.add_row("Genre", f"{cfg.genre}")
            table.add_row("Tone", f"{cfg.tone}")
            table.add_row("Setting", f"{cfg.setting}")
            table.add_row("POV/Tense", f"{cfg.pov} / {cfg.tense}")
            table.add_row("Ending", f"{cfg.ending}")
            table.add_row("Themes", ", ".join(cfg.themes))
            table.add_row("Elements", ", ".join(cfg.required_elements))
            table.add_row("Structure", f"{cfg.narrative_structure} | Arc: {cfg.arc_style}")
            table.add_row("World", f"Time: {cfg.time_period} | Tech: {cfg.tech_level} | Magic: {cfg.magic_system}")
            if cfg.weather_mood:
                table.add_row("Weather", cfg.weather_mood)
            if cfg.cultural_conflict:
                table.add_row("Cultural Conflict", cfg.cultural_conflict)
            prot = f"{cfg.protagonist or 'Alex'} (Want: {cfg.protagonist_want or 'freedom'}, Need: {cfg.protagonist_need or 'connection'}, Flaw: {cfg.protagonist_flaw or 'distrust'})"
            table.add_row("Protagonist", prot)
            if getattr(cfg, 'protagonist_role', None):
                table.add_row("Role/Age", f"{cfg.protagonist_role} | {cfg.protagonist_age}")
            if cfg.ally:
                table.add_row("Ally", f"{cfg.ally} (Trait: {cfg.ally_trait or 'steadfast'}, Secret: {cfg.ally_secret or 'unspoken debt'})")
            _console.print(table)
        else:
            print("\nConfiguration Preview:")
            print(f"Genre: {cfg.genre} | Tone: {cfg.tone} | Setting: {cfg.setting}")
            print(f"POV: {cfg.pov} | Tense: {cfg.tense} | Ending: {cfg.ending}")
            print(f"Themes: {', '.join(cfg.themes)}")
            print(f"Elements: {', '.join(cfg.required_elements)}")
            print(f"Structure: {cfg.narrative_structure} | Arc: {cfg.arc_style}")
            print(f"Time: {cfg.time_period} | Tech: {cfg.tech_level} | Magic: {cfg.magic_system}")
            if cfg.weather_mood:
                print(f"Weather: {cfg.weather_mood}")
            if cfg.cultural_conflict:
                print(f"Cultural Conflict: {cfg.cultural_conflict}")
            print(f"Protagonist: {cfg.protagonist or 'Alex'} (Want: {cfg.protagonist_want or 'freedom'}, Need: {cfg.protagonist_need or 'connection'}, Flaw: {cfg.protagonist_flaw or 'distrust'})")
            if getattr(cfg, 'protagonist_role', None):
                print(f" - Role: {cfg.protagonist_role} | Age: {cfg.protagonist_age}")
            if cfg.ally:
                print(f"Ally: {cfg.ally} (Trait: {cfg.ally_trait or 'steadfast'}, Secret: {cfg.ally_secret or 'unspoken debt'})")
        if getattr(args, 'yes', False):
            ans = 'y'
        else:
            ans = input("Proceed? (y/n) [y]: ") or 'y'
        return ans.lower().startswith('y')

    if not confirm_config(config):
        print("Aborting.")
        return

    # Merge world-pack if provided (load a worlds_<pack>.json and merge into memory)
    if getattr(args, 'world_pack', None):
        try:
            base = os.path.dirname(__file__)
            wpath = os.path.join(base, f"worlds_{args.world_pack}.json")
            if os.path.exists(wpath):
                with open(wpath, 'r', encoding='utf-8') as f:
                    extra = json.load(f)
                # simple deep merge into primary worlds via a temp generator
                tmp = StoryGenerator(StoryConfig(), load_model=False)
                primary = tmp.worlds or {"settings": {}}
                for k, v in (extra.get('settings', {}) or {}).items():
                    primary.setdefault('settings', {})[k] = v
                # write back merged set so subsequent generator sees it
                with open(os.path.join(base, 'worlds.json'), 'w', encoding='utf-8') as wf:
                    json.dump(primary, wf, indent=2)
        except Exception:
            pass

    generator = StoryGenerator(
        config,
        load_model=(getattr(args, 'use_model', False) and not getattr(args, 'no_model', False) and not getattr(args, 'dry_run', False)),
        deterministic=getattr(args, 'deterministic', False),
        dry_run=getattr(args, 'dry_run', False)
    )

    # Optional story-level cache
    cache_hit_text: Optional[str] = None
    story_cache_path = None
    if getattr(args, 'cache_story', False):
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            cfg_key_src = json.dumps({
                'cfg': vars(config),
                'seed': args.seed,
                'output': args.output,
                'multi': args.multi_endings
            }, sort_keys=True).encode('utf-8', errors='ignore')
            cfg_hash = hashlib.md5(cfg_key_src).hexdigest()
            story_cache_path = os.path.join(cache_dir, f"story_{cfg_hash}.txt")
            if os.path.exists(story_cache_path):
                with open(story_cache_path, 'r', encoding='utf-8') as f:
                    cache_hit_text = f.read()
        except Exception:
            cache_hit_text = None

    output_chunks: List[str] = []
    if cache_hit_text is None:
        if args.output in ["story", "both"]:
            if getattr(args, 'step_mode', False):
                # Interactive section-by-section reveal; still assemble final text
                print("\nStep mode: press Enter to reveal each section.")
                sections = [
                    ("Hook", generator.generate_hook()),
                    ("Inciting Incident", generator.generate_inciting_incident()),
                    ("Complications", "\n".join(generator.generate_complications())),
                    ("Midpoint", generator.generate_midpoint()),
                    ("Crisis", generator.generate_crisis()),
                    ("Climax", generator.generate_climax()),
                ]
                story_text = ""
                for label, content in sections:
                    input(f"[Enter] {label}...")
                    print(f"\n{label}:\n{content}\n")
                    story_text += f"\n\n{label}:\n{content}"
                # Endings
                if getattr(args, 'multi_endings', 1) and args.multi_endings > 1:
                    endings = []
                    for et in ["bittersweet", "triumphant", "open-ended"][: args.multi_endings]:
                        endings.append(f"{et.title()} Ending:\n" + generator.generate_resolution())
                    story_text += "\n\n" + "\n\n".join(endings)
                else:
                    story_text += "\n\nResolution:\n" + generator.generate_resolution()
                output_chunks.append("Generated Story:\n")
                output_chunks.append(story_text.strip())
            else:
                output_chunks.append("Generated Story:\n")
                # Respect multi-endings by appending alternates after main story
                main_story = generator.generate_story()
                if getattr(args, 'multi_endings', 1) and args.multi_endings > 1:
                    alt_blocks = []
                    for et in ["bittersweet", "triumphant", "open-ended"][: args.multi_endings]:
                        alt_blocks.append(f"\n\nAlternate {et.title()} Ending:\n" + generator.generate_resolution())
                    main_story = main_story + "".join(alt_blocks)
                output_chunks.append(main_story)
        if args.output in ["outline", "both"]:
            output_chunks.append("\nGenerated Outline:\n")
            for section, content in generator.generate_outline().items():
                output_chunks.append(f"{section}:\n{content}\n")
        output_text = "\n".join(output_chunks)
        # Write to story-level cache if requested
        if getattr(args, 'cache_story', False) and story_cache_path:
            try:
                with open(story_cache_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
            except Exception:
                pass
    else:
        output_text = cache_hit_text

    # If dry-run, print prompts instead of story text
    if getattr(args, 'dry_run', False):
        print("Dry-run mode: prompts that would be sent to the model (per section):\n")
        for entry in getattr(generator, '_prompt_log', []) or []:
            print(f"[{entry.get('section')}] {entry.get('prompt')}")
        # Also show outline if requested
        if args.output in ["outline", "both"]:
            print("\nGenerated Outline (fallbacks only, LM disabled):\n")
            for section, content in generator.generate_outline().items():
                print(f"{section}:\n{content}\n")
        return
    def _export_output(text: str, path: str) -> Tuple[bool, str]:
        """Export text to path by extension. Returns (ok, info_message)."""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in [".txt", ""]:
                with open(path or "output.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                return True, f"Saved TXT to {path}"
            if ext == ".md":
                md = text.replace("Generated Story:", "# Generated Story").replace("Generated Outline:", "\n\n# Generated Outline")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(md)
                return True, f"Saved Markdown to {path}"
            if ext == ".docx":
                try:
                    from docx import Document
                except Exception:
                    # Fallback to txt
                    fallback = os.path.splitext(path)[0] + ".txt"
                    with open(fallback, "w", encoding="utf-8") as f:
                        f.write(text)
                    return False, f"python-docx not installed. Saved TXT instead to {fallback}"
                doc = Document()
                for para in text.split("\n\n"):
                    doc.add_paragraph(para)
                doc.save(path)
                return True, f"Saved DOCX to {path}"
            if ext == ".epub":
                try:
                    from ebooklib import epub
                except Exception:
                    fallback = os.path.splitext(path)[0] + ".txt"
                    with open(fallback, "w", encoding="utf-8") as f:
                        f.write(text)
                    return False, f"ebooklib not installed. Saved TXT instead to {fallback}"
                book = epub.EpubBook()
                book.set_identifier("storygen-epub")
                book.set_title("StoryGen Output")
                book.set_language("en")
                chapter = epub.EpubHtml(title="Story", file_name="story.xhtml", lang="en")
                html = "<h1>StoryGen Output</h1>" + "".join(f"<p>{p}</p>" for p in text.split("\n\n"))
                chapter.set_content(html)
                book.add_item(chapter)
                book.toc = (chapter,)
                book.add_item(epub.EpubNcx())
                book.add_item(epub.EpubNav())
                book.spine = ["nav", chapter]
                epub.write_epub(path, book)
                return True, f"Saved EPUB to {path}"
            # Unknown extension -> write plain
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return True, f"Saved to {path}"
        except Exception as e:
            return False, f"Failed to export to {path}: {e}"

    if getattr(args, 'save', None):
        ok, msg = _export_output(output_text, args.save)
        print(msg)
        if not ok:
            # also print the content for immediate use
            print(output_text)
    else:
        print(output_text)

if __name__ == "__main__":
    main()