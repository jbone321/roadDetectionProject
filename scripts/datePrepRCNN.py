# So faster RCNN model does not have a specific format
# of annotations so COCO and KITTI must match
# COCO is a richer format so we convert KITTI to COCO

import json
import os
from PIL import Image

class KittiCoco:
    def __init__(self, kFolder, iFolder, outFolder):
        # will dynamically add classes based on KITTI labels
        self.classes = []
        self.classToId = {}
        self.coco = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.kFolder = kFolder
        self.iFolder = iFolder
        self.outFolder = outFolder

        self.aId = 1
        self.iId = 1

    def image(self, image):
        iPath = os.path.join(self.iFolder, image + ".png")

        if not os.path.exists(iPath):
            return None
        
        try:
            i = Image.open(iPath)
            w, h = i.size
        except:
            print(f"Corrupted image: {iPath}")
            return None
        
    def processKitti(self, line):
        parts = line.strip().split(" ")

        name = parts[0]
        bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]

        return name, bbox
    
    def convert(self):
        print("Converting KITTI to COCO format...")

        labels = [f for f in os.listdir(self.kFolder) if f.endswith(".txt")]
        labels.sort()

        print(f"{len(labels)} labels found.")

        for f in labels:
            name = f.replace(".txt", ".png")
            info  = self.image(name)

            if info is None:
                print(f"Skipping image {name} due to read error.")
                continue

            w, h = info

            self.coco["images"].append({
                "id": self.iId,
                "file_name": name,
                "width": w,
                "height": h
            })

            lPath = os.path.join(self.kFolder, f)
            with open(lPath, "r") as lf:
                lines = lf.readlines()

            for line in lines:
                obj, bbox = self.processKitti(line)

                if obj not in self.classToId:
                    self.classToId[obj] = len(self.classes) + 1
                    self.classes.append(obj)
                    self.coco["categories"].append({
                        "id": self.classToId[obj],
                        "name": obj
                    })
                    print(f"New class found: {obj}")
            

                #convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
                bboxCOCO = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                area = bboxCOCO[2] * bboxCOCO[3]

                self.coco["annotations"].append({
                    "id": self.aId,
                    "image_id": self.iId,
                    "category_id": self.classToId[obj],
                    "bbox": bboxCOCO,
                    "area": area,
                    "iscrowd": 0
                })

                self.aId += 1
            self.iId += 1

        self.saveJson()
        self.summary()

    def saveJson(self):
        with open(self.outFolder, "w") as f:
            json.dump(self.coco, f)
        print(f"COCO annotations saved to {self.outFolder}")

    def summary(self):
        print("Conversion complete.")
        print(f"Total images: {len(self.coco["images"])}")
        print(f"Total annotations: {len(self.coco["annotations"])}")
        print(f"Total categories: {len(self.coco["categories"])}")
        print("Categories:")
        for category in self.coco["categories"]:
            print(f" - {category["name"]} (ID: {category["id"]})")


# MAIN TO BE BUILT IN COLAB