import traceback, tempfile, os, json
import numpy as np, cv2
from pathlib import Path
from PIL import Image
def case(name, fn):
    try:
        r = fn()
        print(f"[OK] {name}: {repr(r)[:300]}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"[ERR] {name}: {type(e).__name__}: {str(e)[:300]} @ {tb.filename.split('site-packages/')[-1]}:{tb.lineno}")
from ultralytics.data import converter as C, utils as U
from ultralytics.utils import checks
td = Path(tempfile.mkdtemp()); os.chdir(td)
md = td / "masks"; md.mkdir()
m = np.zeros((100, 120), np.uint8); m[10:40, 10:50] = 1; m[60:90, 60:110] = 3; m[0:5, 0:5] = 1
cv2.imwrite(str(md / "a.png"), m)
cv2.imwrite(str(md / "rgb.png"), cv2.merge([m, m, m]))
cv2.imwrite(str(md / "u16.png"), m.astype(np.uint16))
Image.fromarray(m, mode="L").convert("P").save(md / "pal.png")
m2 = np.zeros((100, 120), np.uint8); m2[50, 10:60] = 2
cv2.imwrite(str(md / "line.png"), m2)
case("seg masks convert", lambda: C.convert_segment_masks_to_yolo_seg(str(md), str(td / "out"), classes=3))
for f in sorted((td / "out").glob("*.txt")) if (td / "out").exists() else []:
    print("   ", f.name, [l[:60] for l in f.read_text().splitlines()])
ann = td / "ann"; ann.mkdir()
coco = {"images": [{"id": 1, "file_name": "x.jpg", "height": 100, "width": 200}, {"id": 2, "file_name": "y.jpg", "height": 100, "width": 200}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40], "iscrowd": 0, "segmentation": [[10, 10, 60, 10, 60, 50, 10, 50]], "area": 2000,
                         "keypoints": [20, 20, 2] * 17, "num_keypoints": 17},
                        {"id": 2, "image_id": 1, "category_id": 3, "bbox": [100, 10, 50, 40], "iscrowd": 0, "segmentation": [[100, 10, 150, 10, 150, 50], [120, 60, 130, 60, 130, 70]], "area": 2000},
                        {"id": 3, "image_id": 2, "category_id": 90, "bbox": [0, 0, 0, 0], "iscrowd": 0, "segmentation": [], "area": 0}],
        "categories": [{"id": 1, "name": "person"}]}
json.dump(coco, open(ann / "instances_val.json", "w"))
for kw in [dict(), dict(use_segments=True), dict(use_keypoints=True), dict(cls91to80=False)]:
    case(f"convert_coco {kw}", lambda: C.convert_coco(str(ann), str(td / "coco_out"), **kw))
for f in sorted(td.glob("coco_out*/labels/**/*.txt")):
    print("   ", f.relative_to(td), f.read_text().splitlines())
case("coco91_to_coco80", lambda: len(C.coco91_to_coco80_class()))
case("coco80_to_coco91", lambda: len(C.coco80_to_coco91_class()))
case("merge_multi_segment 3", lambda: [s.shape for s in C.merge_multi_segment([[0, 0, 1, 0, 1, 1], [5, 5, 6, 5, 6, 6], [10, 10, 11, 10, 11, 11]])])
case("merge_multi_segment 2", lambda: [s.shape for s in C.merge_multi_segment([[0, 0, 1, 0, 1, 1], [5, 5, 6, 5, 6, 6]])])
d = td / "DOTA"; (d / "images" / "train").mkdir(parents=True); (d / "images" / "val").mkdir(parents=True)
(d / "labels" / "train_original").mkdir(parents=True); (d / "labels" / "val_original").mkdir(parents=True)
cv2.imwrite(str(d / "images" / "train" / "P1.png"), np.zeros((100, 200, 3), np.uint8))
(d / "labels" / "train_original" / "P1.txt").write_text("imagesource:GoogleEarth\ngsd:0.1\n10 10 50 10 50 40 10 40 plane 0\n10 10 50 10 50 40 10 40 unknown-thing 1\n")
case("dota", lambda: C.convert_dota_to_yolo_obb(str(d)))
print("   ", [p.read_text() for p in (d / "labels" / "train").glob("*.txt")])
case("polygon2mask", lambda: U.polygon2mask((100, 100), [np.array([10, 10, 50, 10, 50, 50, 10, 50], float)]).sum())
case("polygon2mask ds4", lambda: U.polygon2mask((100, 100), [np.array([10, 10, 50, 10, 50, 50, 10, 50], float)], downsample_ratio=4).shape)
case("polygons2masks", lambda: U.polygons2masks((100, 100), [np.array([10, 10, 50, 10, 50, 50, 10, 50], float)], color=1).shape)
case("polygons2masks_overlap empty", lambda: [x.shape for x in U.polygons2masks_overlap((100, 100), [])])
case("img2label_paths", lambda: U.img2label_paths(["/a/images/b/images/c.jpg", "/x/images.d/y.png"]))
case("img2label_paths noimages", lambda: U.img2label_paths(["/a/b/c.jpg"]))
cv2.imwrite("im.jpg", np.zeros((50, 60, 3), np.uint8)); open("im.txt", "w").write("0 0.5 0.5 0.2 0.2\n")
case("compress_one_image", lambda: U.compress_one_image("im.jpg", "im2.jpg"))
case("compress_one_image png", lambda: (cv2.imwrite("rgba.png", np.zeros((50, 60, 4), np.uint8)), U.compress_one_image("rgba.png", "rgba_c.jpg"))[1])
case("exif_size", lambda: U.exif_size(Image.open("im.jpg")))
for lbl in ["0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2", "0 1.5 0.5 0.2 0.2", "-1 0.5 0.5 0.2 0.2", "0 0.1 0.1 0.5 0.1 0.5 0.5", "0 0.5 0.5 0.2", "", "0 0.5 0.5 0.2 0.2 0.3 0.3 2"]:
    open("im.txt", "w").write(lbl + "\n")
    for kp in [False, True]:
        case(f"verify_image_label {lbl!r} kp={kp}", lambda: U.verify_image_label(("im.jpg", "im.txt", "", kp, 1, 2 if not kp else 3, 1, False, 3)))
case("get_hash", lambda: U.get_hash(["im.jpg", "nope"]))
case("find_dataset_yaml", lambda: U.find_dataset_yaml(td))
case("check_imgsz 641", lambda: checks.check_imgsz(641))
case("check_imgsz list", lambda: checks.check_imgsz([641, 480]))
case("check_imgsz str", lambda: checks.check_imgsz("640"))
case("check_imgsz tuple", lambda: checks.check_imgsz((640,)))
case("check_imgsz float", lambda: checks.check_imgsz(640.5))
case("check_imgsz str list", lambda: checks.check_imgsz("[640, 480]"))
case("check_imgsz 0", lambda: checks.check_imgsz(0))
case("check_imgsz min_dim2 int", lambda: checks.check_imgsz(640, min_dim=2))
case("check_imgsz max_dim1 list", lambda: checks.check_imgsz([640, 480], max_dim=1))
case("check_imgsz max_dim1 list1", lambda: checks.check_imgsz([640], max_dim=1))
case("check_version", lambda: checks.check_version("1.10.0", ">=1.9.0,<2"))
case("check_version !=", lambda: checks.check_version("1.10.0", "!=1.10.0"))
case("check_version ==", lambda: checks.check_version("1.10.0", "==1.10"))
case("check_version 2.0.0+cpu", lambda: checks.check_version("2.0.0+cpu", ">=2.0.0"))
case("check_version rc", lambda: checks.check_version("2.1.0rc1", ">=2.1.0"))
case("check_version empty", lambda: checks.check_version("", ">=1"))
case("check_suffix", lambda: checks.check_suffix("a.PT", ".pt"))
case("check_yolov5u_filename", lambda: checks.check_yolov5u_filename("yolov5n.pt"))
case("check_model_file_from_stem", lambda: checks.check_model_file_from_stem("yolo11n"))
case("check_file missing", lambda: checks.check_file("nonexist.yaml"))
from ultralytics.nn.autobackend import check_class_names
case("check_class_names list", lambda: check_class_names(["a", "b"]))
case("check_class_names dict str keys", lambda: check_class_names({"0": "a", "1": "b"}))
case("check_class_names imagenet", lambda: check_class_names({0: "n01440764"})[0])
case("check_class_names gap", lambda: check_class_names({0: "a", 2: "c"}))
