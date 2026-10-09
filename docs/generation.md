# Synthetic data generation

## Setup

| Setting | Value | Why |
| --- | --- | --- |
| Model | Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) | fixed in the project plan; pretrained, no generator training |
| Mode | text-to-image, species name only | no real photo enters the generator, so val/test cannot leak through generation |
| Precision | fp16 on GPU | ~3.5 GB of GPU memory; an 8 GB card is enough |
| Resolution | 512×512 | SD 1.5's training size; other sizes give duplicated or cut-off birds |
| Scheduler | DPM-Solver++, 25 steps | reaches DDIM-50 quality in about half the steps |
| Guidance | 7.5 | SD default; higher gives saturated, less varied images |
| Safety checker | off | it replaces flagged images with black squares; bad images are removed in review |
| File format | JPEG, quality 95 | same as the CUB photos, so the format cannot tell synthetic from real |

### Prompts

`src/prompts.py` turns the class name into the common English name a web caption would contain:
`009.Brewer_Blackbird` → *Brewer's Blackbird*, `110.Geococcyx` → *Greater Roadrunner*,
`141.Artic_Tern` → *Arctic Tern*. Generic rule: a lowercase word joins the previous one with a
hyphen (`Black_and_white_Warbler` → *Black-and-white Warbler*); 26 names that the rule gets
wrong are listed in `NAME_OVERRIDES`. `python src/prompts.py` prints all 200 prompts.

| Template | Prompt |
| --- | --- |
| `photo` | a photo of a {species}, a type of bird |
| `wildlife` | a wildlife photograph of a {species} bird in its natural habitat, sharp focus, high detail |

Negative prompt (both): cartoon, illustration, drawing, painting, 3d render, text, watermark,
frame, multiple birds, blurry, deformed, extra legs, extra wings.

The pilot compares the two templates; the full pool uses the better one for all species.

### Seeds

Image `k` of class `label` uses seed `10000 + 1000·label + k`, drawn on a CPU generator, so it does
not depend on the GPU or the batch size. Consequences:

- any single image can be regenerated;
- pools are nested: 20 per class (2:1 ratio extension) keeps the 10 of the 1:1 pool and adds 10;
- both templates use the same seeds, so the template comparison is paired.

One pool is shared by all classifier seeds, as in the plan: seed variability then reflects the real
subsets and initialization, not differences between generated pools.

## Files

| Path | Content | In git |
| --- | --- | --- |
| `data/synthetic/{pool}/{class_name}/*.jpg` | images | no (shared via Drive) |
| `data/synthetic/{pool}/metadata.csv` | one row per image, written while generating | no |
| `data/synthetic/{pool}/sheets/*.jpg` | contact sheets for review | no (shared via Drive) |
| `splits/manifests/synthetic_{pool}.csv` | manifest: `image_id, label, class_name` + prompt, seed, settings, `gen_time_s`, `device` | yes |
| `reports/generation/review_{pool}.csv` | review decisions | yes |
| `reports/generation/leakage_{pool}.json` | synthetic images matching val/test | yes |

Synthetic manifests start with the same three columns as the real ones, and image ids follow the
same pattern (`synthetic/{pool}/{class_name}/{species}_{index:04d}` vs.
`training/{class_name}/{photo}`), so the mixed-data loader can concatenate a real and a
synthetic manifest. `generate.synthetic_path(image_id)` gives the file.

Pool names: `pilot_{template}` for the pilot, `sd15_{template}` for the full pool.

## Running

On a machine with a CUDA GPU:

```bash
pip install -r requirements.txt
python src/generate.py --pilot                      # 10 species x 5 images -> pilot_photo
python src/generate.py --pilot --template wildlife  # same seeds -> pilot_wildlife
python src/load_data.py                             # real photos for the contact sheets (once)
python src/review_synthetic.py --pool pilot_photo
python src/review_synthetic.py --pool pilot_wildlife
```

The first run downloads the model (~4 GB) into the Hugging Face cache. Generation resumes: if it
stops, re-run the same command and finished images are skipped. A run with a different template,
model, steps or guidance into an existing pool is refused, so one pool never mixes settings.

Full pool, after the acceptance rules below are final:

```bash
python src/generate.py --template photo --per-class 10   # 2,000 images -> sd15_photo
python src/review_synthetic.py --pool sd15_photo
```

`--batch-size 4` is faster on an 8 GB card; the images stay the same up to fp16 noise.

## Pilot

10 species × 5 images per template. Five species are distinctive, five have look-alikes in CUB,
where species fidelity is most at risk:

| Distinctive | Hard |
| --- | --- |
| Northern Cardinal, Blue Jay, Painted Bunting, Brown Pelican, Pileated Woodpecker | California Gull, Herring Gull, Chipping Sparrow, Acadian Flycatcher, Nashville Warbler |

Questions the pilot answers:

1. **Cost:** median seconds per image and the time for 2,000 images (printed by `generate.py`,
   per image in the manifest's `gen_time_s`). Generation time is reported separately from
   training time, with the GPU name.
2. **Species fidelity:** which share of images show the right species, for distinctive vs. hard
   species.
3. **Template:** which template gives more correct and more varied images.
4. **Leakage:** whether SD reproduces val/test photos.

Results: _to fill in after the pilot run._

## Review

`review_synthetic.py` writes, per species, a contact sheet: the top row shows 5 real photos the
classifier trains on (10-shot subset, seed 0; never val/test), the rows below the synthetic images
numbered `#k`. Reviewers fill the `decision` column of `reports/generation/review_{pool}.csv`:

| Decision | Meaning |
| --- | --- |
| `ok` | one bird of the labeled species, photo-like |
| `wrong_species` | a realistic bird, but field marks of another species (bill, wing bars, head pattern, colors) |
| `artifact` | the right species with visible defects: extra or missing limbs, merged birds, garbled feathers |
| `not_a_bird` | no bird, only a part of one, or a drawing / painting |
| `leak` | listed in `leakage_{pool}.json` as a copy of a val/test photo |

Compare against the real row and the species' field marks, not against general bird realism: a
realistic bird of the wrong species is the main risk for training.

### Acceptance rules (draft, to finalize after the pilot)

- An image enters the training pool only with decision `ok`. `leak` is always dropped.
- Dropped images are replaced by the next seeds of the same species (`--per-class` higher), so
  every species keeps exactly 10 synthetic images and the 1:1 ratio holds.
- A species with fewer than 3 of 5 `ok` images in the pilot gets a closer look in the full
  pool; if SD cannot draw it, it gets no synthetic images, and this is reported.
- The unfiltered pool is kept: filtered vs. unfiltered is a planned optional extension.
