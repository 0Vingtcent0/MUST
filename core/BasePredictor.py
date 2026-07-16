import json
import os
import shutil
import time
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import tifffile as tif
import torch
from pytz import timezone

from train_tools.data_utils.transforms import get_pred_transforms


def _normalize_modality(modality):
    """Normalize the modality selector used during prediction."""
    if modality is None:
        return None
    modality = str(modality).strip()
    if modality == "" or modality.lower() in {"multi", "all", "none"}:
        return None
    return modality


class BasePredictor:
    def __init__(
        self,
        model,
        device,
        input_path,
        output_path,
        make_submission=False,
        exp_name=None,
        algo_params=None,
        input_mapping=None,
        root=None,
        modality=None,
    ):
        self.model = model
        self.device = device
        self.input_path = input_path
        self.output_path = output_path
        self.make_submission = make_submission
        self.exp_name = exp_name
        self.input_mapping = input_mapping
        self.root = root
        self.modality = modality

        if algo_params:
            self.__dict__.update((k, v) for k, v in algo_params.items())

        self._setups()

    @torch.no_grad()
    def conduct_prediction(self):
        self.model.to(self.device)
        self.model.eval()
        total_time = 0
        total_times = []

        for pred_item in self.pred_items:
            img_name = pred_item["img_name"]
            img_data = self._get_img_data(pred_item)
            img_data = img_data.to(self.device)

            start = time.time()

            pred_mask = self._inference(img_data)
            pred_mask = self._post_process(pred_mask.squeeze(0).cpu().numpy())
            self.write_pred_mask(
                pred_mask, self.output_path, img_name, self.make_submission
            )

            end = time.time()

            time_cost = end - start
            total_times.append(time_cost)
            total_time += time_cost
            print(
                f"Prediction finished: {img_name}; img size = {img_data.shape}; costing: {time_cost:.2f}s"
            )

        print(f"\n Total Time Cost: {total_time:.2f}s")

        if self.make_submission:
            fname = "%s.zip" % self.exp_name

            os.makedirs("./submissions", exist_ok=True)
            submission_path = os.path.join("./submissions", fname)

            with ZipFile(submission_path, "w") as zipObj2:
                pred_names = sorted(os.listdir(self.output_path))
                for pred_name in pred_names:
                    pred_path = os.path.join(self.output_path, pred_name)
                    zipObj2.write(pred_path)

            print("\n>>>>> Submission file is saved at: %s\n" % submission_path)

        return total_time

    def write_pred_mask(self, pred_mask, output_dir, image_name, submission=False):
        if submission:
            if not (np.max(pred_mask) > 5):
                print("[!Caution] Only %d Cells Detected!!!\n" % np.max(pred_mask))

        file_name = image_name.split(".")[0]
        file_name = file_name + "_label.tiff"
        file_path = os.path.join(output_dir, file_name)

        tif.imwrite(file_path, pred_mask, compression="zlib")

    def _setups(self):
        self.pred_transforms = get_pred_transforms()
        os.makedirs(self.output_path, exist_ok=True)

        for name in os.listdir(self.output_path):
            p = os.path.join(self.output_path, name)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                print(f"[Warn] cannot remove stale item: {p} ({e})")

        now = datetime.now(timezone("Europe/Paris"))
        dt_string = now.strftime("%m%d_%H%M")
        self.exp_name = (
            self.exp_name + dt_string if self.exp_name is not None else dt_string
        )

        self.pred_items = self._load_prediction_items()
        self.img_names = [item["img_name"] for item in self.pred_items]
        print(f"[Prediction] Number of images: {len(self.pred_items)}")

    def _load_prediction_items(self):
        """Load prediction image paths from either a mapping file or an image directory."""
        if self.input_mapping:
            return self._load_prediction_items_from_mapping()

        return [
            {
                "img_name": img_name,
                "img_path": os.path.join(self.input_path, img_name),
                "modality": None,
            }
            for img_name in sorted(os.listdir(self.input_path))
        ]

    def _load_prediction_items_from_mapping(self):
        """Load prediction image paths from a mapping file with optional modality filtering."""
        if not self.root:
            raise ValueError("Prediction from a mapping file requires a dataset root.")
        if not os.path.exists(self.input_mapping):
            raise FileNotFoundError(f"Prediction mapping does not exist: {self.input_mapping}")

        selected_modality = _normalize_modality(self.modality)
        with open(self.input_mapping, "r") as f:
            mapping = json.load(f)

        items = []
        for _, entries in mapping.items():
            for entry in entries:
                entry_modality = entry.get("modality")
                if selected_modality is not None and entry_modality != selected_modality:
                    continue
                rel_path = entry["img"]
                img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.root, rel_path)
                items.append(
                    {
                        "img_name": os.path.basename(rel_path),
                        "img_path": img_path,
                        "modality": entry_modality,
                    }
                )

        if not items:
            raise RuntimeError(
                f"No prediction images found in {self.input_mapping} for modality={self.modality}."
            )
        return sorted(items, key=lambda item: item["img_name"])

    def _get_img_data(self, pred_item):
        img_path = pred_item["img_path"]
        img_data = self.pred_transforms(img_path)
        img_data = img_data.unsqueeze(0)

        return img_data

    def _inference(self, img_data):
        raise NotImplementedError

    def _post_process(self, pred_mask):
        raise NotImplementedError
