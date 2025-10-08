import numpy as np, cv2, argparse, time
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def to_mask(img, idx):
    print(f"[{idx}] → hsv+otsu")
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s_blur = cv2.GaussianBlur(s,(5,5),0)
    v_blur = cv2.GaussianBlur(v,(5,5),0)
    _, s_bin = cv2.threshold(s_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, v_bin = cv2.threshold(v_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bg = cv2.bitwise_and(cv2.bitwise_not(s_bin), v_bin)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, t = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = np.ones((3,3),np.uint8)
    def best_polarity(bw):
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
        n,lab,stat,_=cv2.connectedComponentsWithStats(bw,8,cv2.CV_32S)
        a = 0 if n<=1 else int(stat[1:,cv2.CC_STAT_AREA].max())
        n2,lab2,stat2,_=cv2.connectedComponentsWithStats(cv2.bitwise_not(bw),8,cv2.CV_32S)
        a2 = 0 if n2<=1 else int(stat2[1:,cv2.CC_STAT_AREA].max())
        return bw if a>=a2 else cv2.bitwise_not(bw)
    cand1 = best_polarity(cv2.bitwise_and(t, cv2.bitwise_not(bg)))
    cand2 = best_polarity(t)
    thr = cand1 if np.count_nonzero(cand1)>=np.count_nonzero(cand2) else cand2
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    h0,w0 = thr.shape
    s0 = float(h0*w0)
    fg = float(np.count_nonzero(thr))
    if fg < 0.002*s0 or fg > 0.98*s0:
        print(f"[{idx}] ✖ reject empty/full"); return None
    n,lab,stat,_=cv2.connectedComponentsWithStats(thr,8,cv2.CV_32S)
    if n<=1:
        print(f"[{idx}] ✖ reject no-cc"); return None
    idx2 = 1+np.argmax(stat[1:,cv2.CC_STAT_AREA])
    mask = ((lab==idx2).astype(np.uint8))*255
    print(f"[{idx}] ✓ mask area={int(np.count_nonzero(mask))}")
    return mask

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

def fourier_descriptors(mask, k=20):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.zeros(k, dtype=np.float64)
    cnt = max(cnts, key=cv2.contourArea)
    z = cnt.squeeze(1).astype(np.float64)
    if z.ndim!=2 or z.shape[0] < k+5: 
        return np.zeros(k, dtype=np.float64)
    z = z - z.mean(axis=0)
    c = z[:,0] + 1j*z[:,1]
    C = np.fft.fft(c)
    C = C[1:k+1]
    mag = np.abs(C)
    norm = np.linalg.norm(mag) + 1e-12
    return (mag / norm).astype(np.float64)

def load_trashnet(split, fd_k):
    print(f"🔹 load split='{split}'")
    ds = load_dataset("garythung/trashnet", split=split)
    print("✅ samples:", len(ds))
    X, y = [], []
    t0 = time.time()
    kept = 0
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

def kfold_eval(X,y,k,metric):
    print(f"🔹 5-fold kNN k={k} metric={metric}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    yt, yp = [], []
    fold = 1
    for tr,te in skf.split(Xs,y):
        if metric=="mahalanobis":
            cov = np.cov(Xs[tr].T)
            reg = 1e-3*np.eye(cov.shape[0])
            VI = np.linalg.pinv(cov+reg)
            clf = KNeighborsClassifier(n_neighbors=k, metric="mahalanobis", metric_params={"VI": VI}, weights="distance")
        else:
            clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
        clf.fit(Xs[tr],y[tr])
        p = clf.predict(Xs[te])
        acc = accuracy_score(y[te], p)
        print(f"  ▶ fold {fold} acc={acc:.4f}")
        yt.extend(y[te]); yp.extend(p)
        fold += 1
    acc = accuracy_score(yt,yp)
    cm = confusion_matrix(yt,yp,labels=sorted(np.unique(y)))
    rep = classification_report(yt,yp,digits=3)
    return acc, cm, rep

def main(k, fd_k):
    try:
        X,y,names = load_trashnet("train", fd_k)
    except Exception as e:
        print("⚠️ train split failed, use 'all':", e)
        X,y,names = load_trashnet("all", fd_k)
    if len(X)==0:
        print("❌ no valid samples"); return
    print("🔎 feature dim:", X.shape[1], "samples:", len(X))
    acc_eu, cm_eu, rep_eu = kfold_eval(X,y,k,"euclidean")
    acc_ma, cm_ma, rep_ma = kfold_eval(X,y,k,"mahalanobis")
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
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fd_k", type=int, default=20)
    a = ap.parse_args()
    main(a.k, a.fd_k)
