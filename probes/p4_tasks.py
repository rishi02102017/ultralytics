import traceback
import numpy as np, torch
from pathlib import Path
import cv2, tempfile
def case(name, fn):
    try:
        r = fn()
        print(f"[OK] {name}: {repr(r)[:300]}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:300]} @ {tb.filename.split('site-packages/')[-1]}:{tb.lineno}")
from ultralytics import YOLO
import ultralytics
bus = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
blank = np.zeros((480, 640, 3), np.uint8)
td = Path(tempfile.mkdtemp())
for w in ["yolo11n-seg.pt", "yolo11n-pose.pt", "yolo11n-obb.pt", "yolo11n-cls.pt"]:
    try:
        m = YOLO(w)
    except Exception as e:
        print("[ERR] load", w, e); continue
    for nm, im in [("bus", bus), ("blank", blank)]:
        for rr in [False, True]:
            try:
                r = m.predict(im, verbose=False, retina_masks=rr)[0]
            except Exception as e:
                case(f"{w} {nm} rr={rr} predict", lambda: (_ for _ in ()).throw(e)); continue
            for conv in ["t", "np"]:
                x = r.numpy() if conv == "np" else r
                p = f"{w}/{nm}/rr={rr}/{conv}"
                case(f"{p} summary", lambda: len(x.summary()))
                case(f"{p} to_json", lambda: len(x.to_json()))
                case(f"{p} to_df", lambda: x.to_df().shape)
                case(f"{p} to_csv", lambda: len(x.to_csv()))
                case(f"{p} plot", lambda: x.plot().shape)
                case(f"{p} plot noboxes", lambda: x.plot(boxes=False, masks=True, probs=False, kpt_line=False).shape)
                case(f"{p} save_txt", lambda: x.save_txt(td / f"{w}{nm}{rr}{conv}.txt", save_conf=True))
                case(f"{p} save_crop", lambda: x.save_crop(td / f"c{w}{nm}{rr}{conv}"))
                case(f"{p} verbose", lambda: x.verbose())
                case(f"{p} slice", lambda: len(x[:1]))
