def main():
    config = StoryConfig(
        genre="low fantasy noir",
        tone="gritty",
        pov="first",
        tense="present",
        length="short",
        audience="adult",
        rating="PG-13",
        ending="bittersweet",
        themes=["the cost of memory"],
        required_elements=["lockbox", "rain-slick streets", "stray dog", "nursery rhyme fragment"],
        setting="rain-drenched city, 1920s"
    )
    generator = StoryGenerator(config)
    with open("story.txt", "w") as f:
        f.write("Generated Story:\n\n")
        f.write(generator.generate_story())
    with open("outline.txt", "w") as f:
        f.write("Generated Outline:\n\n")
        for section, content in generator.generate_outline().items():
            f.write(f"{section}:\n{content}\n\n")
    print("Story and outline saved to story.txt and outline.txt")