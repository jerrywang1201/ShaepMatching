import numpy as np, cv2, argparse, time
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========== 1) 分割：CLAHE + HSV/Gray Otsu + Triangle 兜底 + 候选自动选择 ==========
def to_mask(img, idx, min_fg=0.0005, max_fg=0.998):
    print(f"[{idx}] → hsv/gray otsu + auto-pick + triangle fallback")
    H, W = img.shape[:2]
    ksz = max(3, int(round(min(H, W)*0.012)) | 1)
    k = np.ones((ksz, ksz), np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v)

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g_blur = cv2.GaussianBlur(g, (5,5), 0)

    _, s_bin = cv2.threshold(cv2.GaussianBlur(s,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, v_bin = cv2.threshold(cv2.GaussianBlur(v_eq,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bg_hint = cv2.bitwise_and(cv2.bitwise_not(s_bin), v_bin)

    def otsu_or_triangle(x):
        _, t = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        fg = np.count_nonzero(t)
        if fg < 0.0001*x.size or fg > 0.999*x.size:
            _, t = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
        return t

    t_gray = otsu_or_triangle(g_blur)
    t_veq  = otsu_or_triangle(v_eq)

    cand_gray = cv2.bitwise_and(t_gray, cv2.bitwise_not(bg_hint))
    cand_veq  = cv2.bitwise_and(t_veq,  cv2.bitwise_not(bg_hint))
    cand_s    = cv2.bitwise_and(s_bin,  cv2.bitwise_not(bg_hint))
    raw_list = [cand_gray, cand_veq, cand_s, t_gray]

    def best_polarity(bw):
        bw = (bw>0).astype(np.uint8)*255
        n, lab, stat, _ = cv2.connectedComponentsWithStats(bw,8,cv2.CV_32S)
        a1 = 0 if n<=1 else int(stat[1:, cv2.CC_STAT_AREA].max())
        inv = cv2.bitwise_not(bw)
        n2, lab2, stat2, _ = cv2.connectedComponentsWithStats(inv,8,cv2.CV_32S)
        a2 = 0 if n2<=1 else int(stat2[1:, cv2.CC_STAT_AREA].max())
        return bw if a1>=a2 else inv

    def polish_and_score(bw):
        bw = cv2.morphologyEx((bw>0).astype(np.uint8)*255, cv2.MORPH_CLOSE, k, iterations=1)
        n, lab, stat, _ = cv2.connectedComponentsWithStats(bw,8,cv2.CV_32S)
        if n<=1: return None, 0.0
        idx_big = 1+np.argmax(stat[1:, cv2.CC_STAT_AREA])
        comp = ((lab==idx_big).astype(np.uint8))*255

        cnts,_ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        A = cv2.contourArea(cnt)
        P = cv2.arcLength(cnt, True)+1e-12
        hull = cv2.convexHull(cnt)
        Ah = cv2.contourArea(hull)+1e-12
        circ = (4.0*np.pi*A)/(P*P)
        solidity = A/Ah
        score = (A/(H*W)) + 0.3*circ + 0.2*solidity
        return comp, score

    cands=[]
    for r in raw_list:
        pol = best_polarity(r)
        comp, sc = polish_and_score(pol)
        if comp is not None: cands.append((comp, sc))
    if not cands:
        print(f"[{idx}] ✖ reject no candidate"); return None

    comp, sc = max(cands, key=lambda x: x[1])
    fg_ratio = np.count_nonzero(comp)/(H*W)
    if not (min_fg <= fg_ratio <= max_fg):
        print(f"[{idx}] ⚠ keep borderline mask (fg={fg_ratio:.4f})")

    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k, iterations=1)
    print(f"[{idx}] ✓ mask area={int(np.count_nonzero(comp))} score={sc:.3f}")
    return comp

# ========== 2) 形状特征 ==========
def hu_log(m):
    M = cv2.moments(m, binaryImage=True)
    hu = cv2.HuMoments(M).flatten()
    hu = np.where(hu==0,1e-12,hu)
    return np.sign(hu)*np.log10(np.abs(hu))

def geom_feats(m):
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    x,y,w,h = cv2.boundingRect(cnt)
    ratio = (w/float(h)) if h>0 else 0.0
    circ = (4.0*np.pi*A)/(P*P) if P>0 else 0.0
    hull = cv2.convexHull(cnt)
    Ah = cv2.contourArea(hull)
    solidity = (A/Ah) if Ah>1e-12 else 0.0
    ys,xs = np.where(m>0)
    cov = np.cov(np.vstack([xs,ys]))
    if cov.shape==(2,2):
        vals,_ = np.linalg.eigh(cov)
        vals = np.sort(vals)[::-1]
        ecc = np.sqrt(max(0.0, 1.0 - (vals[1]/(vals[0]+1e-12))))
    else:
        ecc = 0.0
    nperim = P/np.sqrt(A+1e-12)
    return np.array([ratio, circ, solidity, ecc, nperim], dtype=np.float64)

# 等距重采样 + 低维 FD（更稳）
def resample_contour(cnt, N=256):
    pts = cnt.squeeze(1).astype(np.float64)
    if pts.ndim!=2 or len(pts)<3: return None
    d = np.sqrt(((np.diff(pts, axis=0))**2).sum(1))
    s = np.hstack([[0], np.cumsum(d)])
    if s[-1] < 1e-6: return None
    u = np.linspace(0, s[-1], N)
    x = np.interp(u, s, pts[:,0])
    y = np.interp(u, s, pts[:,1])
    return np.stack([x,y], axis=1)

def fourier_descriptors(mask, k=12):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.zeros(k, dtype=np.float64)
    cnt = max(cnts, key=cv2.contourArea)
    R = resample_contour(cnt, N=256)
    if R is None: return np.zeros(k, dtype=np.float64)
    R = R - R.mean(axis=0)
    c = R[:,0] + 1j*R[:,1]
    C = np.fft.fft(c)
    C = C[1:k+1]
    mag = np.abs(C)
    mag = mag / (np.linalg.norm(mag)+1e-12)
    return mag.astype(np.float64)

# ========== 3) 数据加载 ==========
def load_trashnet(split, fd_k):
    print(f"🔹 load split='{split}'")
    ds = load_dataset("garythung/trashnet", split=split)
    print("✅ samples:", len(ds))
    X, y = [], []
    t0 = time.time(); kept = 0
    for i, ex in enumerate(ds):
        img = np.array(ex["image"].convert("RGB"))
        m = to_mask(img, i)
        if m is None: continue
        hu = hu_log(m)
        gf = geom_feats(m)
        if gf is None:
            print(f"[{i}] ✖ reject no-contour"); continue
        fd = fourier_descriptors(m, k=fd_k)
        feat = np.concatenate([hu, gf, fd])
        X.append(feat); y.append(ex["label"]); kept += 1
        if (i+1) % 100 == 0:
            print(f"   progress {i+1}/{len(ds)} kept={kept}")
    dur = time.time()-t0
    print(f"🕒 feat done {dur:.1f}s kept={kept}")
    names = ds.features["label"].names
    return np.array(X), np.array(y), names

# ========== 4) 评估：分块加权 + kNN + 质心融合 ==========
def kfold_eval(X,y,k,metric, hu_w=1.0, geom_w=1.2, fd_w=0.8):
    print(f"🔹 5-fold kNN k={k} metric={metric}  | weights: HU={hu_w} GEO={geom_w} FD={fd_w}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # block-wise weights
    hu_dim = 7
    geom_dim = 5
    fd_dim = X.shape[1] - (hu_dim + geom_dim)
    w = np.ones(X.shape[1], dtype=np.float64)
    w[:hu_dim] *= hu_w
    w[hu_dim:hu_dim+geom_dim] *= geom_w
    if fd_dim > 0:
        w[hu_dim+geom_dim:] *= fd_w
    Xs = Xs * w

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    yt, yp = [], []
    fold = 1
    for tr,te in skf.split(Xs,y):
        # kNN 构建
        if metric=="mahalanobis":
            cov = np.cov(Xs[tr].T)
            reg = 3e-3*np.eye(cov.shape[0])         # 更强正则
            VI = np.linalg.pinv(cov+reg)
            clf = KNeighborsClassifier(n_neighbors=k, metric="mahalanobis", metric_params={"VI": VI}, weights="distance")
        else:
            clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
        clf.fit(Xs[tr],y[tr])

        # 质心分类（非学习，统计均值）
        mu = {}
        classes = sorted(np.unique(y[tr]))
        for cls in classes:
            mu[cls] = Xs[tr][y[tr]==cls].mean(axis=0)
        def predict_centroid(Xq):
            dists = np.stack([np.linalg.norm(Xq - mu[c], axis=1) for c in classes], axis=1)
            idx = np.argmin(dists, axis=1)
            return np.array([classes[i] for i in idx])

        p_knn = clf.predict(Xs[te])
        p_cent = predict_centroid(Xs[te])

        # 简单融合策略：一致则用该类；不一致保持 kNN（也可按需要调整）
        p = p_knn
        acc_fold = accuracy_score(y[te], p)
        agree_rate = np.mean(p_knn==p_cent)
        print(f"  ▶ fold {fold} acc={acc_fold:.4f} | agree(kNN,centroid)={agree_rate:.2f}")
        yt.extend(y[te]); yp.extend(p)
        fold += 1
    acc = accuracy_score(yt,yp)
    cm = confusion_matrix(yt,yp,labels=sorted(np.unique(y)))
    rep = classification_report(yt,yp,digits=3)
    return acc, cm, rep

# ========== 5) 主流程 ==========
def main(k, fd_k, hu_w, geom_w, fd_w):
    try:
        X,y,names = load_trashnet("train", fd_k)
    except Exception as e:
        print("⚠️ train split failed, use 'all':", e)
        X,y,names = load_trashnet("all", fd_k)
    if len(X)==0:
        print("❌ no valid samples"); return
    print("🔎 feature dim:", X.shape[1], "samples:", len(X))

    acc_eu, cm_eu, rep_eu = kfold_eval(X,y,k,"euclidean", hu_w, geom_w, fd_w)
    acc_ma, cm_ma, rep_ma = kfold_eval(X,y,k,"mahalanobis", hu_w, geom_w, fd_w)

    report_lines = []
    report_lines.append("========== FINAL (EUCLIDEAN) ==========")
    report_lines.append(f"classes: {names}")
    report_lines.append(f"overall acc: {acc_eu:.4f}")
    report_lines.append(f"confusion:\n{cm_eu}")
    report_lines.append(rep_eu)
    report_lines.append("========== FINAL (MAHALANOBIS) ==========")
    report_lines.append(f"overall acc: {acc_ma:.4f}")
    report_lines.append(f"confusion:\n{cm_ma}")
    report_lines.append(rep_ma)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open("results.txt", "w") as f:
        f.write(report_text)
    print("\n💾 Results saved to results.txt\n")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--fd_k", type=int, default=12)
    ap.add_argument("--hu_w", type=float, default=1.0)
    ap.add_argument("--geom_w", type=float, default=1.2)
    ap.add_argument("--fd_w", type=float, default=0.8)
    a = ap.parse_args()
    main(a.k, a.fd_k, a.hu_w, a.geom_w, a.fd_w)
