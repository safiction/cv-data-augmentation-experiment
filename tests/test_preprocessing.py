"""Transforms and loaders: shapes, determinism, and the pairing guarantees the experiment relies on."""
import pytest
import torch
from PIL import Image

from conftest import requires_data
from load_data import load_manifest
from preprocessing import (CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, eval_transform, get_dataset, get_loader,
                           to_rgb, train_aug_transform)


def bird_like_image(size=(500, 375)):
    """Deterministic non-square RGB image with structure, so crops and flips change pixels."""
    x = torch.linspace(0, 1, size[0]).expand(size[1], -1)
    y = torch.linspace(0, 1, size[1]).unsqueeze(1).expand(-1, size[0])
    rgb = torch.stack([x, y, x * y]) * 255
    return Image.fromarray(rgb.permute(1, 2, 0).to(torch.uint8).numpy())


@pytest.mark.parametrize("mode", ["L", "P", "RGBA", "CMYK", "RGB"])
def test_to_rgb(mode):
    img = to_rgb(bird_like_image().convert(mode))
    assert img.mode == "RGB"


def test_eval_transform_shape_and_determinism():
    img = bird_like_image()
    a, b = eval_transform()(img), eval_transform()(img)
    assert a.shape == (3, CROP_SIZE, CROP_SIZE) and a.dtype == torch.float32
    assert torch.equal(a, b)


def test_eval_transform_normalizes_with_imagenet_stats():
    gray = Image.new("RGB", (300, 300), (124, 116, 104))  # ImageNet mean in 0-255
    out = eval_transform()(gray)
    expected = [(v / 255 - m) / s for v, m, s in zip((124, 116, 104), IMAGENET_MEAN, IMAGENET_STD)]
    assert torch.allclose(out.mean(dim=(1, 2)), torch.tensor(expected), atol=1e-6)


def test_train_aug_transform_is_random_but_seedable():
    img, aug = bird_like_image(), train_aug_transform()
    torch.manual_seed(0)
    a = [aug(img) for _ in range(5)]
    torch.manual_seed(0)
    b = [aug(img) for _ in range(5)]
    assert all(x.shape == (3, CROP_SIZE, CROP_SIZE) for x in a)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(a[0], x) for x in a[1:])


def test_augment_only_for_train():
    with pytest.raises(ValueError):
        get_dataset("val", augment=True)
    with pytest.raises(ValueError):
        get_dataset("test", k_shot=10)


def first_batches(loader, n=2):
    it = iter(loader)
    return [next(it) for _ in range(n)]


@requires_data
@pytest.mark.parametrize("split, k_shot", [("train", 10), ("val", None), ("test", None)])
def test_dataset_follows_manifest(split, k_shot):
    ds = get_dataset(split, k_shot=k_shot, seed=0)
    manifest = load_manifest(split, k_shot, 0 if k_shot else None)
    # with_format(None) drops the transform, so reading the ids does not decode every image
    assert ds.with_format(None)["image_id"] == manifest["image_id"].tolist()
    item = ds[0]
    assert item["pixel_values"].shape == (3, CROP_SIZE, CROP_SIZE) and item["label"] == manifest["label"][0]


@requires_data
def test_eval_loader_unshuffled_and_deterministic():
    a = first_batches(get_loader("val", batch_size=16, num_workers=0))
    b = first_batches(get_loader("val", batch_size=16, num_workers=0))
    manifest = load_manifest("val")
    assert torch.cat([x["label"] for x in a]).tolist() == manifest["label"].tolist()[:32]
    assert sum((x["image_id"] for x in a), []) == manifest["image_id"].tolist()[:32]
    assert all(torch.equal(x["pixel_values"], y["pixel_values"]) for x, y in zip(a, b))


@requires_data
@pytest.mark.parametrize("num_workers", [0, 2])
def test_train_loader_reproducible_and_paired_across_arms(num_workers):
    def batches(augment, seed=0):
        torch.manual_seed(seed)
        return first_batches(get_loader("train", k_shot=10, seed=seed, augment=augment, batch_size=32,
                                        num_workers=num_workers))

    aug, aug_again, plain, other_seed = batches(True), batches(True), batches(False), batches(False, seed=1)
    assert all(torch.equal(x["pixel_values"], y["pixel_values"]) for x, y in zip(aug, aug_again))
    # The augmented pool is twice as large; its shuffled order differs from the baseline.
    manifest = load_manifest("train", 10, 0)
    labels = dict(zip(manifest["image_id"], manifest["label"]))
    for batch in aug:
        assert batch["label"].tolist() == [labels[image_id] for image_id in batch["image_id"]]
    assert all(x["image_id"] == y["image_id"] for x, y in zip(aug, aug_again))
    assert not torch.equal(plain[0]["label"], other_seed[0]["label"])


@requires_data
def test_train_loader_covers_subset_each_epoch_in_new_order():
    loader = get_loader("train", k_shot=5, seed=0, num_workers=0)
    epochs = [list(loader.sampler) for _ in range(2)]  # row indices only, no image decoding
    assert sorted(epochs[0]) == list(range(len(loader.dataset))) == sorted(epochs[1])
    assert epochs[0] != epochs[1]
