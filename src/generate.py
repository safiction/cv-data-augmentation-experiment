"""Generate the synthetic training pool with Stable Diffusion 1.5, text-to-image only.

Only species names go into the generator, no real photos, so val/test cannot reach the pool through
generation. SD was trained on web images and may still reproduce CUB photos; review_synthetic.py
checks the pool against val/test with the detectors from leakage.py.

Each image has its own seed, SEED_OFFSET + 1000 * label + index, drawn on a CPU generator so it
does not depend on the GPU model or the batch size. Any image can be regenerated alone, and pools
are nested like the k-shot subsets: --per-class 20 keeps the 10 images of --per-class 10 and adds
10 more. The script resumes: images already in the pool's metadata are skipped, so it can be
stopped at any time and restarted with the same command.

Images are saved as JPEG (quality 95) like the CUB photos, so the file format is not a cue the
classifier could use to tell synthetic images from real ones. The safety checker is off: it
replaces flagged images with black squares; bad images are removed during review instead.

Output:
  data/synthetic/{pool}/{class_name}/{species}_{index:04d}.jpg   images (not in git; shared via Drive)
  data/synthetic/{pool}/metadata.csv                              one row per image, appended live
  splits/manifests/synthetic_{pool}.csv                           manifest + metadata (in git)

Manifest image_ids look like "synthetic/{pool}/{class_name}/{species}_{index:04d}", with the same
columns as the real manifests first (image_id, label, class_name); `synthetic_path` gives the file.

Usage (on a CUDA GPU; 8 GB is enough):
  python src/generate.py --pilot                       # 10 species x 5 images, template "photo"
  python src/generate.py --pilot --template wildlife
  python src/generate.py --per-class 10                # full pool: 200 species x 10 = 2,000 images
"""
import argparse
import platform
import time

import pandas as pd
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from load_data import DATA_DIR, load_splits, manifest_path
from prompts import NEGATIVE_PROMPT, TEMPLATES, prompt_for

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
SEED_OFFSET = 10_000
STEPS = 25  # DPM-Solver++ reaches DDIM-50 quality in 20-25 steps
GUIDANCE = 7.5
SIZE = 512  # SD 1.5's training resolution; other sizes produce duplicated or cut-off birds
JPEG_QUALITY = 95

# Pilot: 5 distinctive species and 5 that are hard to tell from look-alikes in CUB
# (two gulls, a sparrow, an Empidonax flycatcher, a warbler).
PILOT_CLASSES = [
    "017.Cardinal", "073.Blue_Jay", "016.Painted_Bunting", "100.Brown_Pelican", "188.Pileated_Woodpecker",
    "059.California_Gull", "062.Herring_Gull", "116.Chipping_Sparrow", "037.Acadian_Flycatcher",
    "172.Nashville_Warbler",
]
PILOT_PER_CLASS = 5

# Settings that must stay the same inside one pool; a resumed run with others is refused.
POOL_SETTINGS = ["template", "model", "steps", "guidance"]


def seed_for(label, index):
    return SEED_OFFSET + 1000 * label + index


def image_id_for(pool, class_name, index):
    return f"synthetic/{pool}/{class_name}/{class_name.split('.', 1)[1]}_{index:04d}"


def synthetic_path(image_id):
    """File of a synthetic image_id from a synthetic manifest."""
    return DATA_DIR / f"{image_id}.jpg"


def load_pipeline(model_id=MODEL_ID):
    """SD 1.5 in fp16 on the GPU (~3.5 GB of memory), DPM-Solver++ scheduler, no safety checker."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def device_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else f"cpu ({platform.processor()})"


def generate(pool, class_names, per_class, template, batch_size=1, model_id=MODEL_ID,
             steps=STEPS, guidance=GUIDANCE):
    """Generate the missing images of a pool and append their rows to its metadata.csv."""
    labels = dict(load_splits()[["class_name", "label"]].drop_duplicates().to_numpy())
    meta_path = SYNTHETIC_DIR / pool / "metadata.csv"
    settings = {"template": template, "model": model_id, "steps": steps, "guidance": guidance}

    done = set()
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        for key in POOL_SETTINGS:
            if set(meta[key]) != {settings[key]}:
                raise ValueError(f"pool {pool} was generated with {key}={sorted(set(meta[key]))}, "
                                 f"not {settings[key]}; use another --pool")
        done = {i for i in meta["image_id"] if synthetic_path(i).exists()}

    todo = [(labels[c], c, k) for c in class_names for k in range(per_class)
            if image_id_for(pool, c, k) not in done]
    print(f"{pool}: {len(done)} images done, {len(todo)} to generate on {device_name()}")
    if not todo:
        return

    pipe = load_pipeline(model_id)
    gpu = device_name()
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        prompts = [prompt_for(c, template) for _, c, _ in batch]
        seeds = [seed_for(label, k) for label, _, k in batch]

        t0 = time.perf_counter()
        images = pipe(
            prompts,
            negative_prompt=[NEGATIVE_PROMPT] * len(batch),
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=SIZE,
            width=SIZE,
            generator=[torch.Generator("cpu").manual_seed(s) for s in seeds],
        ).images
        seconds = (time.perf_counter() - t0) / len(batch)

        rows = []
        for (label, class_name, k), prompt, seed, img in zip(batch, prompts, seeds, images):
            image_id = image_id_for(pool, class_name, k)
            path = synthetic_path(image_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, quality=JPEG_QUALITY)
            rows.append({
                "image_id": image_id, "label": label, "class_name": class_name,
                "prompt": prompt, "negative_prompt": NEGATIVE_PROMPT, "seed": seed, **settings,
                "scheduler": type(pipe.scheduler).__name__, "size": SIZE, "batch_size": batch_size,
                "gen_time_s": round(seconds, 3), "device": gpu,
            })
        pd.DataFrame(rows).to_csv(meta_path, mode="a", header=not meta_path.exists(), index=False)
        print(f"  {start + len(batch)}/{len(todo)}  {seconds:.2f} s/image")


def write_manifest(pool):
    """splits/manifests/synthetic_{pool}.csv from the pool's metadata, one row per image on disk."""
    meta = pd.read_csv(SYNTHETIC_DIR / pool / "metadata.csv")
    meta = meta.drop_duplicates("image_id", keep="last")
    meta = meta[[synthetic_path(i).exists() for i in meta["image_id"]]]
    path = manifest_path(f"synthetic_{pool}")
    meta.sort_values("image_id").to_csv(path, index=False)
    return meta, path


def report_cost(meta, full_pool=2000):
    """Median, not mean: the first batch includes CUDA warm-up."""
    per_image = meta["gen_time_s"].median()
    print(f"{len(meta)} images on {meta['device'].iloc[-1]}: median {per_image:.2f} s/image, "
          f"total {meta['gen_time_s'].sum() / 60:.1f} min; "
          f"{full_pool} images would take ~{per_image * full_pool / 3600:.1f} h")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true", help=f"{len(PILOT_CLASSES)} pilot species only")
    parser.add_argument("--per-class", type=int, help=f"images per species (default {PILOT_PER_CLASS} "
                                                      "for --pilot, else 10)")
    parser.add_argument("--template", default="photo", choices=sorted(TEMPLATES))
    parser.add_argument("--pool", help="output name (default pilot_{template} or sd15_{template})")
    parser.add_argument("--batch-size", type=int, default=1, help="images per forward pass; 1 fits any 6 GB GPU")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--model", default=MODEL_ID)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    class_names = PILOT_CLASSES if args.pilot else sorted(load_splits()["class_name"].unique())
    per_class = args.per_class or (PILOT_PER_CLASS if args.pilot else 10)
    pool = args.pool or f"{'pilot' if args.pilot else 'sd15'}_{args.template}"

    generate(pool, class_names, per_class, args.template, args.batch_size, args.model,
             args.steps, args.guidance)
    meta, path = write_manifest(pool)
    report_cost(meta)
    print("Manifest:", path.relative_to(path.parents[2]))
