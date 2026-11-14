import traceback, tempfile, os, json
import numpy as np, torch
from pathlib import Path
from ultralytics.engine.results import Results, Boxes, OBB, Keypoints, Probs, Masks

def case(name, fn):
    try:
        r = fn()
        s = repr(r)
        print(f"[OK] {name}: {s[:300]}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:300]} @ {tb.filename.split('site-packages/')[-1]}:{tb.lineno}")

img = np.zeros((480, 640, 3), np.uint8)
names = {0: "a", 1: "b", 2: "c"}
tmp = Path(tempfile.mkdtemp())
boxes = torch.tensor([[10., 20, 100, 200, 0.9, 0], [50, 60, 300, 400, 0.5, 2]])
tboxes = torch.tensor([[10., 20, 100, 200, 7, 0.9, 0], [50, 60, 300, 400, 8, 0.5, 2]])
obb = torch.tensor([[100., 100, 50, 30, 0.3, 0.9, 1]])
kpts = torch.rand(2, 17, 3) * 100
masks = torch.zeros(2, 480, 640); masks[0, 20:200, 10:100] = 1; masks[1, 60:400, 50:300] = 1
empty = torch.zeros(0, 6)

variants = {
  "det": dict(boxes=boxes),
  "det_track": dict(boxes=tboxes),
  "det_empty": dict(boxes=empty),
  "obb": dict(obb=obb),
  "obb_empty": dict(obb=torch.zeros(0, 7)),
  "pose": dict(boxes=boxes, keypoints=kpts),
  "pose_xy": dict(boxes=boxes, keypoints=kpts[..., :2]),
  "pose_empty": dict(boxes=empty, keypoints=torch.zeros(0, 17, 3)),
  "seg": dict(boxes=boxes, masks=masks),
  "seg_empty": dict(boxes=empty, masks=torch.zeros(0, 480, 640)),
  "cls": dict(probs=torch.tensor([0.1, 0.7, 0.2])),
  "none": dict(),
}
for vname, kw in variants.items():
    for conv in ["torch", "numpy"]:
        r = Results(img, path="x.jpg", names=names, **kw)
        if conv == "numpy":
            r = r.numpy()
        n = f"{vname}/{conv}"
        case(f"{n} len", lambda: len(r))
        case(f"{n} verbose", lambda: r.verbose())
        case(f"{n} summary", lambda: r.summary())
        case(f"{n} summary_norm", lambda: r.summary(normalize=True))
        case(f"{n} to_json", lambda: r.to_json()[:150])
        case(f"{n} to_df", lambda: r.to_df().shape)
        case(f"{n} to_csv", lambda: r.to_csv()[:150])
        case(f"{n} to_dict", lambda: r.to_dict() if hasattr(r, 'to_dict') else None)
        case(f"{n} plot", lambda: r.plot().shape)
        case(f"{n} plot_noconf", lambda: r.plot(conf=False, labels=False, boxes=False).shape)
        case(f"{n} plot_pil", lambda: type(r.plot(pil=True)))
        case(f"{n} plot_instance", lambda: r.plot(color_mode="instance").shape)
        case(f"{n} save_txt", lambda: open(r.save_txt(tmp / f"{vname}_{conv}.txt", save_conf=True)).read()[:150] if (tmp / f"{vname}_{conv}.txt").exists() or True else None)
        case(f"{n} save_crop", lambda: r.save_crop(tmp / f"crops_{vname}_{conv}"))
        case(f"{n} getitem0", lambda: len(r[0]))
        case(f"{n} getitem_slice", lambda: len(r[0:1]))
        case(f"{n} getitem_boolmask", lambda: len(r[np.array([True] * len(r), dtype=bool)]) if len(r) else 'skip')
        case(f"{n} cpu", lambda: len(r.cpu()))
        case(f"{n} to_float16", lambda: len(r.to(dtype=torch.float16)))

b = Boxes(boxes.numpy(), (480, 640))
case("Boxes numpy xywhn", lambda: b.xywhn)
case("Boxes numpy xyxyn", lambda: b.xyxyn)
case("Boxes 1d", lambda: Boxes(boxes[0], (480, 640)).xywh)
case("Boxes 1d xywhn", lambda: Boxes(boxes[0], (480, 640)).xywhn)
case("Boxes 1d id", lambda: Boxes(tboxes[0], (480, 640)).id)
case("OBB 1d", lambda: OBB(obb[0], (480, 640)).xyxyxyxy)
case("OBB numpy xyxy", lambda: OBB(obb.numpy(), (480, 640)).xyxy)
case("OBB numpy xyxyxyxyn", lambda: OBB(obb.numpy(), (480, 640)).xyxyxyxyn)
case("OBB empty xyxy", lambda: OBB(torch.zeros(0, 7), (480, 640)).xyxy)
case("OBB empty np xyxy", lambda: OBB(np.zeros((0, 7)), (480, 640)).xyxy)
case("Kpt numpy xyn", lambda: Keypoints(kpts.numpy(), (480, 640)).xyn.shape)
case("Kpt numpy conf", lambda: Keypoints(kpts.numpy(), (480, 640)).conf.shape)
case("Kpt 2d", lambda: Keypoints(kpts[0], (480, 640)).xy.shape)
case("Probs numpy top5", lambda: Probs(np.array([0.1, 0.7, 0.2])).top5)
case("Probs numpy top5conf", lambda: Probs(np.array([0.1, 0.7, 0.2])).top5conf)
case("Probs top1conf np", lambda: Probs(np.array([0.1, 0.7, 0.2])).top1conf)
case("Masks numpy xy", lambda: [m.shape for m in Masks(masks.numpy(), (480, 640)).xy])
case("Masks numpy xyn", lambda: [m.shape for m in Masks(masks.numpy(), (480, 640)).xyn])
case("Masks empty xy", lambda: Masks(torch.zeros(0, 480, 640), (480, 640)).xy)
case("Masks lowres xy", lambda: [m.max(0) for m in Masks(masks[:, ::2, ::2].contiguous(), (480, 640)).xy])
r = Results(img, path="x.jpg", names=names, boxes=boxes)
case("update boxes", lambda: (r.update(boxes=torch.tensor([[-5., -5, 700, 500, .9, 0]])), r.boxes.xyxy)[1])
case("update numpy boxes", lambda: (r.update(boxes=np.array([[-5., -5, 700, 500, .9, 0]])), r.boxes.xyxy)[1])
case("new", lambda: r.new().boxes)
case("show none", lambda: None)
case("save", lambda: r.save(str(tmp / "a.jpg")))
case("save_txt dir creation", lambda: r.save_txt(tmp / "deep" / "dir" / "a.txt"))
case("summary decimals", lambda: r.summary(decimals=1))
case("to_json normalize", lambda: r.to_json(normalize=True)[:200])
case("to_df normalize", lambda: r.to_df(normalize=True))
case("to_csv normalize", lambda: r.to_csv(normalize=True, decimals=2))
case("to_sql?", lambda: hasattr(r, "to_sql"))
