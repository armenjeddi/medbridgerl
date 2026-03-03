import argparse, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def random_subset(ds, n: int, seed: int = 42):
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def collate_fn(processor, batch):
    imgs, ys = zip(*batch)  # imgs are PIL
    x = processor(images=list(imgs), return_tensors="pt")
    return x["pixel_values"], x["image_grid_thw"], torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def extract_feats(visual, loader, device: str, pooling: str):
    feats_all, y_all = [], []
    visual.eval()

    for pixel_values, grid_thw, y in tqdm(loader, desc="extract", dynamic_ncols=True):
        pixel_values = pixel_values.to(device, non_blocking=True).to(
            dtype=getattr(visual, "dtype", pixel_values.dtype)
        )
        grid_thw = grid_thw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = visual(pixel_values, grid_thw=grid_thw)
        tokens = out[0] if isinstance(out, (tuple, list)) else out

        b = grid_thw.size(0)
        tok_per_img = int(grid_thw[0].prod().item())
        tokens = tokens.view(b, tok_per_img, -1)

        feats = tokens.mean(1) if pooling == "mean" else tokens.max(1).values
        feats = F.normalize(feats.float(), dim=-1)

        feats_all.append(feats.cpu())
        y_all.append(y.cpu())

    return torch.cat(feats_all, 0), torch.cat(y_all, 0)


@torch.no_grad()
def knn_top1_top5(
    train_feats, train_labels, val_feats, val_labels, k: int, chunk: int = 128
):
    train_feats_t = train_feats.t()  # (D, Ntrain)
    top1 = 0
    top5 = 0
    n = val_feats.size(0)

    for i in tqdm(range(0, n, chunk), desc="knn", dynamic_ncols=True):
        v = val_feats[i : i + chunk]
        y = val_labels[i : i + chunk]

        sims = v @ train_feats_t
        nn_idx = sims.topk(k, dim=1).indices
        nn_labels = train_labels[nn_idx]  # (B, k)

        # top-1 majority
        pred1 = []
        for r in range(nn_labels.size(0)):
            pred1.append(torch.bincount(nn_labels[r]).argmax())
        pred1 = torch.stack(pred1)
        top1 += (pred1 == y).sum().item()

        # top-5 most common labels among k
        for r in range(nn_labels.size(0)):
            counts = torch.bincount(nn_labels[r])
            top5_labels = torch.topk(counts, k=min(5, counts.numel())).indices
            top5 += int((y[r] == top5_labels).any().item())

    return top1 / n, top5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/datasets/imagenet")
    ap.add_argument("--base_vlm_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    device = args.device
    processor = AutoProcessor.from_pretrained(args.base_vlm_id).image_processor

    tfm_train = T.Compose(
        [
            T.RandomResizedCrop(
                args.image_size, interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(),
        ]
    )
    tfm_val = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(args.image_size),
        ]
    )

    train_ds = datasets.ImageFolder(f"{args.data_root}/train", transform=tfm_val)
    val_ds = datasets.ImageFolder(f"{args.data_root}/val", transform=tfm_val)

    train_ds = random_subset(train_ds, args.limit_train, seed=0)
    val_ds = random_subset(val_ds, args.limit_val, seed=1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(processor, b),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(processor, b),
    )

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_vlm_id,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    visual = vlm.model.visual.to(device).eval()

    train_feats, train_labels = extract_feats(
        visual, train_loader, device, args.pooling
    )
    val_feats, val_labels = extract_feats(visual, val_loader, device, args.pooling)

    top1, top5 = knn_top1_top5(
        train_feats,
        train_labels.long(),
        val_feats,
        val_labels.long(),
        k=args.k,
        chunk=args.chunk,
    )
    print(f"kNN (cosine)  k={args.k}  top1={top1*100:.2f}%  top5={top5*100:.2f}%")
    print(
        f"train={len(train_ds)}  val={len(val_ds)}  image_size={args.image_size}  pooling={args.pooling}"
    )


if __name__ == "__main__":
    main()


## torchrun --nnodes=1 --nproc_per_node=1 knn.py
