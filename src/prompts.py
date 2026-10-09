"""Text prompts for the generator, built from CUB class names.

CUB folder names drop apostrophes and hyphens ("009.Brewer_Blackbird", "022.Chuck_will_Widow"),
and a few use a genus, a short or a misspelled name ("110.Geococcyx", "017.Cardinal",
"141.Artic_Tern"). Stable Diffusion's text encoder (CLIP) learned bird names from web captions,
so prompts use the common English name a caption would contain.

Prompts stay short: CLIP reads at most 77 tokens and silently drops the rest.
"""

# Names the generic rule below gets wrong: possessives, genus names, old or short names.
NAME_OVERRIDES = {
    "009.Brewer_Blackbird": "Brewer's Blackbird",
    "017.Cardinal": "Northern Cardinal",
    "022.Chuck_will_Widow": "Chuck-will's-widow",
    "023.Brandt_Cormorant": "Brandt's Cormorant",
    "061.Heermann_Gull": "Heermann's Gull",
    "067.Anna_Hummingbird": "Anna's Hummingbird",
    "074.Florida_Jay": "Florida Scrub-Jay",
    "091.Mockingbird": "Northern Mockingbird",
    "092.Nighthawk": "Common Nighthawk",
    "093.Clark_Nutcracker": "Clark's Nutcracker",
    "098.Scott_Oriole": "Scott's Oriole",
    "103.Sayornis": "phoebe (Sayornis)",
    "105.Whip_poor_Will": "Eastern Whip-poor-will",
    "110.Geococcyx": "Greater Roadrunner",
    "113.Baird_Sparrow": "Baird's Sparrow",
    "115.Brewer_Sparrow": "Brewer's Sparrow",
    "122.Harris_Sparrow": "Harris's Sparrow",
    "123.Henslow_Sparrow": "Henslow's Sparrow",
    "124.Le_Conte_Sparrow": "LeConte's Sparrow",
    "125.Lincoln_Sparrow": "Lincoln's Sparrow",
    "126.Nelson_Sharp_tailed_Sparrow": "Nelson's Sparrow",
    "141.Artic_Tern": "Arctic Tern",
    "146.Forsters_Tern": "Forster's Tern",
    "178.Swainson_Warbler": "Swainson's Warbler",
    "180.Wilson_Warbler": "Wilson's Warbler",
    "193.Bewick_Wren": "Bewick's Wren",
}

# The pilot compares templates; the full pool uses one of them for all 200 species.
TEMPLATES = {
    "photo": "a photo of a {species}, a type of bird",
    "wildlife": "a wildlife photograph of a {species} bird in its natural habitat, sharp focus, high detail",
}

NEGATIVE_PROMPT = ("cartoon, illustration, drawing, painting, 3d render, text, watermark, frame, "
                   "multiple birds, blurry, deformed, extra legs, extra wings")


def species_name(class_name):
    """'001.Black_footed_Albatross' -> 'Black-footed Albatross'.

    A lowercase word continues a compound with the previous word ("Black_and_white" ->
    "Black-and-white"); a capitalized word starts a new word.
    """
    if class_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[class_name]
    words = []
    for token in class_name.split(".", 1)[1].split("_"):
        if words and token.islower():
            words[-1] += "-" + token
        else:
            words.append(token)
    return " ".join(words)


def prompt_for(class_name, template):
    return TEMPLATES[template].format(species=species_name(class_name))


if __name__ == "__main__":
    from load_data import load_splits

    for class_name in sorted(load_splits()["class_name"].unique()):
        print(f"{class_name:40s} {prompt_for(class_name, 'photo')}")
