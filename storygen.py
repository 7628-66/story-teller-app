#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Optional pretty CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


@dataclass
class StoryConfig:
    genre: str = "speculative fiction"
    tone: str = "hopeful"
    pov: str = "third"
    tense: str = "past"
    length: str = "short"
    audience: str = "adult"
    rating: str = "PG-13"
    ending: str = "bittersweet"
    themes: List[str] = field(default_factory=lambda: ["redemption"])
    required_elements: List[str] = field(default_factory=list)
    setting: str = "generic modern world"

    # Optional characters
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
        self.themes = self.themes or ["redemption"]
        self.required_elements = self.required_elements or []


class StoryGenerator:
    def __init__(self, config: StoryConfig, load_model: bool = True, deterministic: bool = False):
        self.config = config
        self.deterministic = deterministic
        self.word_counts = {
            "flash": (300, 700),
            "short": (1000, 2000),
            "novelette": (7000, 12000),
            "chapter": (2000, 4000)
        }
        self.characters = {
            "protagonist": {
                "name": config.protagonist or "Alex",
                "want": config.protagonist_want or "freedom",
                "need": config.protagonist_need or "connection",
                "flaw": config.protagonist_flaw or "distrust",
            },
            "antagonist": {
                "name": config.antagonist or "Rival",
                "goal": config.antagonist_goal or "control",
            },
            "ally": {
                "name": config.ally or "",
                "trait": config.ally_trait or "steadfast",
                "secret": config.ally_secret or "unspoken debt"
            },
        }
        self.templates = self._load_templates()
        self.worlds = self._load_worlds()
        self.generator = None
        self._gen_kwargs = None
        self._allow_model = bool(load_model)

    # ----------------- TEMPLATES -----------------
    def _load_templates(self) -> Dict:
        base = os.path.dirname(__file__)
        path = os.path.join(base, 'templates.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"hook": [], "complication": [], "filler": []}

    # ----------------- WORLD LOADING -----------------
    def _load_worlds(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, 'worlds.json')
        if not os.path.exists(path):
            return {"settings": {}}
        with open(path, 'r', encoding='utf-8') as f:
            worlds = json.load(f)
        for _, details in worlds.get('settings', {}).items():
            details.setdefault('factions', [])
            details.setdefault('artifacts', [])
            details.setdefault('culture', {
                "customs": [], "values": [], "conflicts": [],
                "cuisine": [], "folklore": [], "festivals": []
            })
        return worlds

    # ----------------- TEXT CLEANING -----------------
    def _clean(self, text: str) -> str:
        text = text.replace('\n', ' ').strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        kept = [s for s in sentences if len(s) > 10]
        return ' '.join(kept)[:500]

    # ----------------- STORY SECTIONS -----------------
    def generate_hook(self):
        if self.templates.get("hook"):
            tmpl = random.choice(self.templates["hook"])
            verb = "stands" if self.config.tense == "present" else "stood"
            return tmpl.format(
                setting=self.config.setting,
                protagonist=self.characters["protagonist"]["name"],
                element=random.choice(self.config.required_elements or ["mysterious object"]),
                theme=self.config.themes[0],
                verb=verb
            )
        return f"In {self.config.setting}, {self.characters['protagonist']['name']} stood at the edge of change."

    def generate_inciting_incident(self):
        return f"When {self.characters['protagonist']['name']} discovers a {random.choice(self.config.required_elements or ['mysterious object'])}, their journey begins."

    def generate_complications(self):
        results = []
        if self.templates.get("complication"):
            for i in range(2):
                tmpl = random.choice(self.templates["complication"])
                results.append(tmpl.format(
                    protagonist=self.characters["protagonist"]["name"],
                    antagonist=self.characters["antagonist"]["name"],
                    element=random.choice(self.config.required_elements or ["secret"]),
                    flaw=self.characters["protagonist"]["flaw"],
                    theme=self.config.themes[0],
                    setting=self.config.setting,
                    ally=self.characters["ally"]["name"] or "an ally"
                ))
        else:
            results = [
                f"{self.characters['protagonist']['name']}'s flaw ({self.characters['protagonist']['flaw']}) complicates their choices.",
                f"{self.characters['antagonist']['name']} advances their plan: {self.characters['antagonist']['goal']}."
            ]
        return results

    def generate_midpoint(self):
        return f"At the midpoint, {self.characters['protagonist']['name']} realizes the true cost of {self.characters['protagonist']['want']} and glimpses their deeper need: {self.characters['protagonist']['need']}."

    def generate_crisis(self):
        return f"A crisis strikes: {self.config.setting} trembles, forcing {self.characters['protagonist']['name']} to choose between their flaw and their need."

    def generate_climax(self):
        return f"In the climax, {self.characters['protagonist']['name']} confronts {self.characters['antagonist']['name']} with {random.choice(self.config.required_elements or ['courage'])}."

    def generate_resolution(self):
        endings = {
            "bittersweet": f"{self.characters['protagonist']['name']} wins, but at a painful cost, embodying {self.config.themes[0]}.",
            "triumphant": f"{self.characters['protagonist']['name']} emerges victorious, carrying {self.config.themes[0]} into the future.",
            "open-ended": f"The tale closes on uncertainty, as {self.characters['protagonist']['name']} stares into an unknown tomorrow."
        }
        return endings.get(self.config.ending, endings["bittersweet"])

    # ----------------- FULL STORY -----------------
    def generate_story(self):
        word_target = random.randint(*self.word_counts[self.config.length])
        sections = [
            self.generate_hook(),
            self.generate_inciting_incident(),
            "\n".join(self.generate_complications()),
            self.generate_midpoint(),
            self.generate_crisis(),
            self.generate_climax(),
            self.generate_resolution()
        ]
        story = "\n\n".join(sections)

        fillers = self.templates.get("filler", [])
        while len(story.split()) < word_target:
            if fillers:
                tmpl = random.choice(fillers)
                story += " " + tmpl.format(
                    setting=self.config.setting,
                    theme=self.config.themes[0],
                    protagonist=self.characters["protagonist"]["name"],
                    element=random.choice(self.config.required_elements or ["object"])
                )
            else:
                story += f" The {self.config.setting} echoed with {self.config.themes[0]}."
        return " ".join(story.split()[:word_target])


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Story Generator")
    p.add_argument("--length", choices=["flash", "short", "novelette", "chapter"], default="short")
    p.add_argument("--themes", type=str, help="Comma themes")
    p.add_argument("--elements", type=str, help="Comma required elements")
    p.add_argument("--ending", choices=["bittersweet", "triumphant", "open-ended"], default="bittersweet")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = StoryConfig(
        length=args.length,
        themes=[t.strip() for t in (args.themes or "redemption").split(",")],
        required_elements=[e.strip() for e in (args.elements or "").split(",") if e.strip()],
        ending=args.ending
    )
    gen = StoryGenerator(cfg)
    story = gen.generate_story()

    if console:
        console.print(Panel(story, title="Generated Story", subtitle=f"{cfg.genre}, {cfg.tone}"))
    else:
        print(story)


if __name__ == "__main__":
    main()
