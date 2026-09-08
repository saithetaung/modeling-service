# service.py
import bentoml
from PIL import Image as PILImage
from ultralytics import YOLO

PROD_MODEL_TAG = "oil_spill_detector_v2:n7qt7e4mucuznqfi"      # 89 mAP — paste your actual tag
FALLBACK_MODEL_TAG = "oil_spill_detector_v1:n4jt43emucujpqfi"  # 77 mAP — paste your actual tag


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 60})
class OilSpillDetector:

    def __init__(self):
        prod_path = bentoml.models.get(PROD_MODEL_TAG).path_of("model.pt")
        fallback_path = bentoml.models.get(FALLBACK_MODEL_TAG).path_of("model.pt")

        # Native Ultralytics loading — no torch.load, no pickle, no Runner.
        self.prod_model = YOLO(prod_path)
        self.fallback_model = YOLO(fallback_path)

    def _format_results(self, results) -> dict:
        r = results[0]
        return {
            "boxes": r.boxes.xyxy.tolist() if r.boxes is not None else [],
            "classes": r.boxes.cls.tolist() if r.boxes is not None else [],
            "confidences": r.boxes.conf.tolist() if r.boxes is not None else [],
            "masks": [m.tolist() for m in r.masks.xy] if r.masks is not None else [],
        }

    @bentoml.api
    def detect(self, image: PILImage.Image, use_fallback: bool = False) -> dict:
        try:
            model = self.fallback_model if use_fallback else self.prod_model
            results = model.predict(image, verbose=False)
            label = "fallback (77 mAP)" if use_fallback else "production (89 mAP)"
            return {"model": label, **self._format_results(results)}
        except Exception as e:
            if not use_fallback:
                results = self.fallback_model.predict(image, verbose=False)
                return {"model": "fallback (auto-recovered)", **self._format_results(results)}
            raise e

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok"}