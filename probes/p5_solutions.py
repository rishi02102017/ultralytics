import traceback, tempfile, os, json
import numpy as np, torch, cv2
from pathlib import Path
import ultralytics
from ultralytics import solutions
def case(name, fn):
    try:
        r = fn()
        print(f"[OK] {name}: {repr(r)[:250]}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:300]} @ {tb.filename.split('site-packages/')[-1]}:{tb.lineno}")
bus = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
blank = np.zeros((480, 640, 3), np.uint8)
td = Path(tempfile.mkdtemp()); os.chdir(td)
json.dump([{"points": [[0, 0], [300, 0], [300, 300], [0, 300]]}], open("park.json", "w"))
specs = {
 "ObjectCounter": dict(region=[(20, 400), (1000, 400)]),
 "ObjectCounter_poly": dict(region=[(20, 400), (1000, 400), (1000, 800), (20, 800)]),
 "Heatmap": dict(colormap=cv2.COLORMAP_PARULA),
 "Heatmap_region": dict(colormap=cv2.COLORMAP_PARULA, region=[(20, 400), (1000, 400)]),
 "SpeedEstimator": dict(),
 "DistanceCalculation": dict(),
 "QueueManager": dict(region=[(20, 400), (1000, 400), (1000, 800), (20, 800)]),
 "Analytics_line": dict(analytics_type="line"),
 "Analytics_bar": dict(analytics_type="bar"),
 "Analytics_pie": dict(analytics_type="pie"),
 "Analytics_area": dict(analytics_type="area"),
 "AIGym": dict(model="yolo11n-pose.pt"),
 "TrackZone": dict(region=[(20, 400), (1000, 400), (1000, 800), (20, 800)]),
 "VisionEye": dict(),
 "InstanceSegmentation": dict(model="yolo11n-seg.pt"),
 "ObjectBlurrer": dict(),
 "ObjectCropper": dict(crop_dir=str(td / "crops")),
 "RegionCounter": dict(region={"a": [(20, 400), (1000, 400), (1000, 800), (20, 800)], "b": [(0, 0), (100, 0), (100, 100)]}),
 "SecurityAlarm": dict(),
 "ParkingManagement": dict(json_file="park.json"),
}
for nm, kw in specs.items():
    cls = getattr(solutions, nm.split("_")[0])
    for model in ([kw.get("model", "yolo11n.pt")] + (["yolo11n-obb.pt"] if "model" not in kw else [])):
        for tracker in (["tracktrack.yaml", None] if nm not in ("ObjectCropper",) else [None]):
            k = dict(kw, model=model, show=False, verbose=False)
            if tracker is None: k["tracker"] = None
            tag = f"{nm} {model} trk={tracker}"
            try:
                s = cls(**k)
            except Exception as e:
                case(f"{tag} init", lambda: (_ for _ in ()).throw(e)); continue
            for fi, im in enumerate([bus, blank, bus, bus[::-1].copy(), blank]):
                if nm.startswith("Analytics"):
                    case(f"{tag} f{fi}", lambda: str(s(im.copy(), fi))[:120])
                else:
                    case(f"{tag} f{fi}", lambda: str(s(im.copy()))[:120])
