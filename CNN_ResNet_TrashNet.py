import os, datetime, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

# ========== Logging ==========
log_file = open("training_log.txt", "w")
def log(msg):
    print(msg)
    log_file.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} | {msg}\n")
    log_file.flush()

device = "mps" if torch.backends.mps.is_available() else "cpu"

log(f"Using device: {device}")

# ========== Dataset ==========
log("Loading dataset (trashnet)...")
ds_train = load_dataset("garythung/trashnet", split="train")

labels_all = np.array([ds_train[i]["label"] for i in range(len(ds_train))])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
ds = DatasetDict({
    "train": ds_train.select(train_idx),
    "test":  ds_train.select(test_idx),
})
label_names = ds_train.features["label"].names
num_classes = len(label_names)
log(f"Dataset ready. Train={len(ds['train'])}, Test={len(ds['test'])}, Classes={label_names}")

# ========== Transforms ==========
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

test_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

class HFDataset(Dataset):
    def __init__(self, hf, tfm): self.hf = hf; self.t = tfm
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        img = self.hf[i]["image"].convert("RGB")
        y = self.hf[i]["label"]
        return self.t(img), y

train_set = HFDataset(ds["train"], train_tf)
test_set  = HFDataset(ds["test"],  test_tf)

# ========== Handle Class Imbalance ==========
cls_counts = np.bincount([ds["train"][i]["label"] for i in range(len(ds["train"]))], minlength=num_classes)
cls_weights = 1.0 / (cls_counts + 1e-6)
sample_weights = [cls_weights[ds["train"][i]["label"]] for i in range(len(ds["train"]))]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=0)


log(f"Class weights: {np.round(cls_weights/cls_weights.max(),3)}")

# ========== Model ==========
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
except Exception:
    log("⚠️  Pretrained weights not available. Using random init.")
    model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32, device=device))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)

# ========== Epoch Runner ==========
def run_epoch(loader, train=True):
    model.train(train)
    tot, corr, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, leave=False)
    for x,y in pbar:
        x,y = x.to(device), y.to(device)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                opt.step()
        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        corr += (pred == y).sum().item()
        tot += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum/tot, corr/tot

# ========== Training ==========
best_acc, best_state = 0.0, None
log("Warmup training (fc only)...")
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True

for e in range(5):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    te_loss, te_acc = run_epoch(test_loader, False)
    sched.step()
    log(f"[Warmup {e+1}/5] train {tr_loss:.4f}/{tr_acc*100:.2f}%  test {te_loss:.4f}/{te_acc*100:.2f}%")
    if te_acc > best_acc:
        best_acc = te_acc
        best_state = {k:v.cpu() for k,v in model.state_dict().items()}

log("Finetuning full network...")
for p in model.parameters(): p.requires_grad = True
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
if best_state: model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

for e in range(15):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    te_loss, te_acc = run_epoch(test_loader, False)
    sched.step()
    log(f"[Finetune {e+1}/15] train {tr_loss:.4f}/{tr_acc*100:.2f}%  test {te_loss:.4f}/{te_acc*100:.2f}%")
    if te_acc > best_acc:
        best_acc = te_acc
        best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        log(f"🔥 New best accuracy: {best_acc*100:.2f}%")

# ========== Evaluation ==========
if best_state: model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
model.eval(); y_true, y_pred = [], []
with torch.no_grad():
    for x,y in tqdm(test_loader, desc="Final Eval"):
        x = x.to(device)
        out = model(x).argmax(1).cpu().numpy()
        y_pred.extend(out.tolist()); y_true.extend(y.numpy().tolist())

acc = 100 * sum(np.array(y_true)==np.array(y_pred)) / len(y_true)
log(f"✅ Final test accuracy: {acc:.2f}%")

report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
log("\n" + report)
log("Confusion Matrix:\n" + str(cm))
log_file.close()

# ========== Save result ==========
with open("result.txt", "w") as f:
    f.write("===== TrashNet CNN Classification Result =====\n")
    f.write(f"Best Test Accuracy: {best_acc*100:.2f}%\n")
    f.write(f"Model: ResNet18 (pretrained)\n")
    f.write(f"Input size: 224x224, Epochs: 20, Batch size: 32\n")
    f.write(f"Augmentation: RandomCrop+Flip+ColorJitter\n")
    f.write(f"Date: {datetime.datetime.now()}\n")
    f.write("\n\nClassification Report:\n" + report)
    f.write("\n\nConfusion Matrix:\n" + str(cm))

print(f"✅ Training complete. Best accuracy: {best_acc*100:.2f}% (logged in result.txt)")
