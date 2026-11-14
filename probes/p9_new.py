import traceback, tempfile, os
import numpy as np, torch, cv2
from pathlib import Path
import ultralytics
from ultralytics import YOLO
def case(name, fn):
    try:
        r = fn()
        print(f"[OK] {name}: {repr(r)[:300]}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:300]} @ {tb.filename.split('site-packages/')[-1]}:{tb.lineno}")
td = Path(tempfile.mkdtemp()); os.chdir(td)
bus = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
big = cv2.resize(bus, (2430, 3240))
wide = np.concatenate([bus, bus, bus], 1)
small = cv2.resize(bus, (200, 300))
for w in ["yolo26n.pt", "yolo26n-seg.pt", "yolo26n-pose.pt", "yolo26n-obb.pt", "yolo26n-sem.pt", "yolo26n-depth.pt", "yolo26n-cls.pt"]:
    try:
        m = YOLO(w)
    except Exception as e:
        print("[ERR] load", w, e); continue
    for nm, im in [("bus", bus), ("big", big), ("wide", wide), ("small", small)]:
        for kw in [dict(), dict(tile=True), dict(tile=320), dict(tile=True, retina_masks=True), dict(tile=640, imgsz=320)]:
            tag = f"{w} {nm} {kw}"
            try:
                r = m.predict(im, verbose=False, **kw)[0]
            except Exception as e:
                case(f"{tag} predict", lambda: (_ for _ in ()).throw(e)); continue
            case(f"{tag} len/verbose", lambda: (len(r), r.verbose()))
            if nm != "bus" and kw: continue
            for conv in ["t", "np"]:
                x = r.numpy() if conv == "np" else r
                p = f"{tag} {conv}"
                case(f"{p} summary", lambda: str(x.summary())[:150])
                case(f"{p} to_df", lambda: x.to_df().shape)
                case(f"{p} to_json", lambda: len(x.to_json()))
                case(f"{p} plot", lambda: x.plot().shape)
                case(f"{p} save_txt", lambda: x.save_txt(td / f"{w}{nm}{conv}.txt"))
                case(f"{p} save_crop", lambda: x.save_crop(td / f"c{w}{nm}{conv}"))
                case(f"{p} [0]", lambda: len(x[0]))
                case(f"{p} [:0]", lambda: (len(x[:0]), x[:0].verbose()))
                case(f"{p} [:0].plot", lambda: x[:0].plot().shape)
                case(f"{p} [:0].summary", lambda: x[:0].summary())
                case(f"{p} bool idx", lambda: len(x[np.array([False] * max(len(x), 1))]) if len(x) else None)
    case(f"{w} predict list tile", lambda: [len(r) for r in m.predict([bus, wide], tile=True, verbose=False)])
    case(f"{w} predict batch tile save_txt", lambda: [len(r) for r in m.predict([bus, wide], tile=True, verbose=False, save=True, save_txt=True, project=str(td / "runs"), name=w, exist_ok=True)])
    case(f"{w} track tile", lambda: [len(r) for r in m.track([bus, bus], tile=True, verbose=False)])
from ultralytics.data.utils import tile_windows, save_depth_png, load_depth
case("tile_windows small", lambda: tile_windows((100, 100), 640))
case("tile_windows exact", lambda: tile_windows((640, 1152), 640))
case("tile_windows tuple", lambda: tile_windows((1000, 1000), (320, 640)))
case("depth roundtrip", lambda: (save_depth_png("d.png", np.array([[1.234, 0], [np.nan, 65.0]])), load_depth("d.png"))[1])
case("depth roundtrip 3d", lambda: (save_depth_png("d2.png", np.ones((1, 4, 1))), load_depth("d2.png").shape)[1])
case("depth roundtrip 1px", lambda: (save_depth_png("d3.png", np.ones((1, 1))), load_depth("d3.png").shape)[1])
case("val sem", lambda: YOLO("yolo26n-sem.pt").val(data="cityscapes8.yaml", imgsz=320, plots=True, verbose=False).results_dict)
case("val depth", lambda: YOLO("yolo26n-depth.pt").val(data="depth8.yaml", imgsz=320, plots=True, verbose=False).results_dict)
case("val sem to_df", lambda: YOLO("yolo26n-sem.pt").val(data="cityscapes8.yaml", imgsz=320, plots=False, verbose=False).to_df().shape)
case("val depth to_df", lambda: YOLO("yolo26n-depth.pt").val(data="depth8.yaml", imgsz=320, plots=False, verbose=False).to_df().shape)
case("val det tile", lambda: YOLO("yolo26n.pt").val(data="coco8.yaml", imgsz=320, tile=True, plots=False, verbose=False).box.map50)
