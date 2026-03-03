import os, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms as T
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm


def ddp_init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def collate_fn(processor, batch):
    imgs, ys = zip(*batch)  # imgs are PIL
    x = processor(images=list(imgs), return_tensors="pt")
    return x["pixel_values"], x["image_grid_thw"], torch.tensor(ys, dtype=torch.long)


class LinearProbe(nn.Module):
    def __init__(self, visual, num_classes=1000, pooling="mean", dim=3584):
        super().__init__()
        self.visual = visual
        self.pooling = pooling
        self.dim = dim
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.dim, affine=False, eps=1e-6),
            nn.Linear(self.dim, num_classes),
        )

    @torch.no_grad()
    def encode(self, pixel_values, grid_thw):
        out = self.visual(pixel_values, grid_thw=grid_thw)
        tokens = out[0] if isinstance(out, (tuple, list)) else out
        b = grid_thw.size(0)
        tok = int(grid_thw[0].prod().item()) // 4
        tokens = tokens.view(b, tok, self.dim)
        if self.pooling == "mean":
            return tokens.mean(1)
        return tokens.max(1).values

    def forward(self, pixel_values, grid_thw):
        feats = self.encode(pixel_values, grid_thw).float()
        return self.head(feats)


@torch.no_grad()
def accuracy(logits, y, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        out.append(correct[:k].reshape(-1).float().sum())
    return out


def all_reduce_sum(x):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/datasets/imagenet")
    ap.add_argument("--base_vlm_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--num_classes", type=int, default=1000)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()

    rank, local_rank, world = ddp_init()
    device = torch.device("cuda", local_rank)

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

    train_ds = datasets.ImageFolder(f"{args.data_root}/train", transform=tfm_train)
    val_ds = datasets.ImageFolder(f"{args.data_root}/val", transform=tfm_val)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(processor, b),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(processor, b),
    )

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_vlm_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    visual = vlm.model.visual.to(device).eval()
    for p in visual.parameters():
        p.requires_grad = False

    model = LinearProbe(visual, num_classes=args.num_classes, pooling=args.pooling).to(
        device
    )
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    opt = torch.optim.SGD(
        model.module.head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        tot, c1, c5, loss_sum = 0, 0.0, 0.0, 0.0

        it = train_loader
        if rank == 0:
            it = tqdm(train_loader, desc=f"train e{epoch}", dynamic_ncols=True)

        run_loss = 0.0
        run_n = 0

        for step, (pixel_values, grid_thw, y) in enumerate(it):
            pixel_values = pixel_values.to(device, non_blocking=True).to(
                dtype=visual.dtype
            )
            grid_thw = grid_thw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(pixel_values, grid_thw)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            a1, a5 = accuracy(logits, y, topk=(1, 5))
            bs = y.size(0)

            tot += bs
            c1 += a1.item()
            c5 += a5.item()
            loss_sum += loss.item() * bs

            run_loss += loss.item() * bs
            run_n += bs

            if rank == 0 and (step + 1) % args.log_every == 0:
                avg_loss = run_loss / run_n
                avg_top1 = 100.0 * (c1 / tot)
                avg_top5 = 100.0 * (c5 / tot)
                it.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    top1=f"{avg_top1:.2f}",
                    top5=f"{avg_top5:.2f}",
                )

        x = torch.tensor([tot, c1, c5, loss_sum], device=device, dtype=torch.float64)
        x = all_reduce_sum(x)
        if rank == 0:
            tot_g, c1_g, c5_g, ls_g = x.tolist()
            print(
                f"epoch {epoch} train  loss={ls_g/tot_g:.4f}  top1={100*c1_g/tot_g:.2f}  top5={100*c5_g/tot_g:.2f}"
            )

        torch.cuda.empty_cache()

        model.eval()
        tot, c1, c5, loss_sum = 0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for pixel_values, grid_thw, y in val_loader:
                pixel_values = pixel_values.to(device, non_blocking=True).to(
                    dtype=visual.dtype
                )
                grid_thw = grid_thw.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(pixel_values, grid_thw)
                loss = F.cross_entropy(logits, y)

                a1, a5 = accuracy(logits, y, topk=(1, 5))
                bs = y.size(0)
                tot += bs
                c1 += a1.item()
                c5 += a5.item()
                loss_sum += loss.item() * bs

        x = torch.tensor([tot, c1, c5, loss_sum], device=device, dtype=torch.float64)
        x = all_reduce_sum(x)
        if rank == 0:
            tot_g, c1_g, c5_g, ls_g = x.tolist()
            print(
                f"epoch {epoch} val    loss={ls_g/tot_g:.4f}  top1={100*c1_g/tot_g:.2f}  top5={100*c5_g/tot_g:.2f}"
            )

        sched.step()

        if rank == 0 and args.save:
            torch.save(
                {
                    "head": model.module.head.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                args.save,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
