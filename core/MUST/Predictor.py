import torch
from core.BasePredictor import BasePredictor
from core.MUST.inference_utils import get_mask_output, postprocess_prediction_mask, run_sliding_window

__all__ = ["Predictor"]


class Predictor(BasePredictor):
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
        super(Predictor, self).__init__(
            model,
            device,
            input_path,
            output_path,
            make_submission,
            exp_name,
            algo_params,
            input_mapping=input_mapping,
            root=root,
            modality=modality,
        )

        self.hflip_tta = HorizontalFlip()
        self.vflip_tta = VerticalFlip()
        self.use_tta = algo_params.get("use_tta", False) if algo_params is not None else False

    @torch.no_grad()
    def _inference(self, img_data):
        """Run model inference, optionally with horizontal and vertical flip TTA."""
        img_data = img_data.to(self.device)
        img_base = img_data
        outputs_base = get_mask_output(self._window_inference(img_base))
        outputs_base = outputs_base.cpu().squeeze()
        img_base.cpu()

        if not self.use_tta:
            pred_mask = outputs_base
            return pred_mask

        else:
            img_hflip = self.hflip_tta.apply_aug_image(img_data, apply=True)
            outputs_hflip = get_mask_output(self._window_inference(img_hflip))
            outputs_hflip = self.hflip_tta.apply_deaug_mask(outputs_hflip, apply=True)
            outputs_hflip = outputs_hflip.cpu().squeeze()
            img_hflip.cpu()

            img_vflip = self.vflip_tta.apply_aug_image(img_data, apply=True)
            outputs_vflip = get_mask_output(self._window_inference(img_vflip))
            outputs_vflip = self.vflip_tta.apply_deaug_mask(outputs_vflip, apply=True)
            outputs_vflip = outputs_vflip.cpu().squeeze()
            img_vflip.cpu()

            outputs_hflip[1] = -outputs_hflip[1]
            outputs_vflip[0] = -outputs_vflip[0]

            pred_mask = (outputs_base + outputs_hflip + outputs_vflip) / 3.0

        return pred_mask

    def _window_inference(self, img_data, aux=False):
        return run_sliding_window(
            images=img_data,
            predictor=self.model if not aux else self.model_aux,
            roi_size=512,
            sw_batch_size=4,
            padding_mode="constant",
            mode="constant",
            overlap=0.6,
        )

    def _post_process(self, pred_mask):
        """Generate cell instance masks."""
        return postprocess_prediction_mask(
            pred_mask=pred_mask,
            device=self.device,
            max_pixels_no_pad=5000 * 5000,
            big_chunk=2000,
            use_gpu=True,
            compute_kwargs={"flow_threshold": 0.4, "cellprob_threshold": 0.5},
            strict_less_than_limit=True,
            print_large_message=True,
            print_grid_progress=True,
        )





class HorizontalFlip:
    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            return image.flip(3)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            return mask.flip(3)
        return mask


class VerticalFlip:
    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            return image.flip(2)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            return mask.flip(2)
        return mask
