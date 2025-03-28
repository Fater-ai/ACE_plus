# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torchvision.transforms as T
import numpy as np
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import Config
from PIL import Image


def edit_preprocess(processor, device, edit_image, edit_mask):
    if edit_image is None or processor is None:
        return edit_image
    processor = Config(cfg_dict=processor, load=False)
    processor = ANNOTATORS.build(processor).to(device)
    new_edit_image = processor(np.asarray(edit_image))
    processor = processor.to("cpu")
    del processor
    new_edit_image = Image.fromarray(new_edit_image)
    return Image.composite(new_edit_image, edit_image, edit_mask)

class ACEPlusImageProcessor():
    def __init__(self, max_aspect_ratio=4, d=16, max_seq_len=1024):
        self.max_aspect_ratio = max_aspect_ratio
        self.d = d
        self.max_seq_len = max_seq_len
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def image_check(self, image):
        if image is None:
            return image
        # preprocess
        W, H = image.size
        if H / W > self.max_aspect_ratio:
            image = T.CenterCrop([int(self.max_aspect_ratio * W), W])(image)
        elif W / H > self.max_aspect_ratio:
            image = T.CenterCrop([H, int(self.max_aspect_ratio * H)])(image)
        return self.transforms(image)


    def preprocess(self,
                   reference_image=None,
                   edit_image=None,
                   edit_mask=None,
                   height=1024,
                   width=1024,
                   repainting_scale = 1.0,
                   keep_pixels = False,
                   keep_pixels_rate = 0.8,
                   use_change = False):
        reference_image = self.image_check(reference_image)
        edit_image = self.image_check(edit_image)
        # for reference generation
        if edit_image is None:
            edit_image = torch.zeros([3, height, width])
            edit_mask = torch.ones([1, height, width])
        else:
            if edit_mask is None:
                _, eH, eW = edit_image.shape
                edit_mask = np.ones((eH, eW))
            else:
                edit_mask = np.asarray(edit_mask)
                edit_mask = np.where(edit_mask > 128, 1, 0)
            edit_mask = edit_mask.astype(
                np.float32) if np.any(edit_mask) else np.ones_like(edit_mask).astype(
                np.float32)
            edit_mask = torch.tensor(edit_mask).unsqueeze(0)

        edit_image = edit_image * (1 - edit_mask * repainting_scale)


        out_h, out_w = edit_image.shape[-2:]

        assert edit_mask is not None
        if reference_image is not None:
            _, H, W = reference_image.shape # Original ref dims C, H, W
            _, eH, eW = edit_image.shape    # Edit image dims C, H, W

            grid_size = 16
            processed_ref_image = None # Variable to hold the final reference image to be concatenated

            if not keep_pixels:
                # --- Branch 1: Resize ref to match edit height, align width to grid ---
                target_h = eH
                scale_ratio = target_h / H
                ideal_w = W * scale_ratio

                # Round width DOWN to nearest grid multiple
                target_w = int(ideal_w // grid_size) * grid_size
                target_w = max(0, target_w) # Ensure non-negative

                if target_w > 0:
                    # Resize using target height and grid-aligned width
                    processed_ref_image = T.Resize(
                        (target_h, target_w),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True
                    )(reference_image)
                # If target_w is 0, processed_ref_image remains None

            else: # keep_pixels == True
                # --- Branch 2: Scale ref based on rate, align width to grid, pad height ---
                target_h_scaled = H
                target_w_scaled = W
                # Apply scaling if ref height is too large relative to edit height
                if H >= keep_pixels_rate * eH:
                    target_h_scaled = int(eH * keep_pixels_rate)
                    scale = target_h_scaled / H
                    target_w_scaled = W * scale # Ideal width after height scaling

                # Round scaled width DOWN to nearest grid multiple
                target_w_grid = int(target_w_scaled // grid_size) * grid_size
                target_w_grid = max(0, target_w_grid)

                if target_w_grid > 0:
                    # Resize to the potentially rate-limited height and the grid-aligned width
                    resized_ref = T.Resize(
                        (target_h_scaled, target_w_grid),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True
                    )(reference_image)

                    # Pad vertically (at the bottom) to match the full edit image height (eH)
                    current_h, _ = resized_ref.shape[-2:] # We only need current height for padding calculation
                    delta_h = eH - current_h

                    if delta_h < 0: # Safeguard: if somehow taller after resize, crop
                         print(f"Warning: Reference image taller ({current_h}) than target ({eH}) after resize in keep_pixels branch. Cropping.")
                         processed_ref_image = resized_ref[:, :eH, :]
                    elif delta_h > 0: # Pad if shorter
                         # Padding format: (left, top, right, bottom)
                         padding = (0, 0, 0, delta_h)
                         processed_ref_image = T.Pad(padding, fill=0, padding_mode="constant")(resized_ref)
                    else: # Height already matches eH
                         processed_ref_image = resized_ref
                # If target_w_grid is 0, processed_ref_image remains None

            # --- Concatenation and setting slice_w (common logic after branching) ---
            if processed_ref_image is not None:
                 # Final check: Ensure height matches edit_image height exactly before concat.
                 # This should generally be true due to the logic above, but acts as a safeguard.
                 if processed_ref_image.shape[-2] != eH:
                     print(f"Warning: Height mismatch ({processed_ref_image.shape[-2]} vs {eH}) before final concat. Resizing ref height.")
                     processed_ref_image = T.Resize((eH, processed_ref_image.shape[-1]), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(processed_ref_image)

                 # Concatenate the processed reference image to the left of the edit image
                 edit_image = torch.cat([processed_ref_image, edit_image], dim=-1) # dim=-1 is width

                 # Create a corresponding mask of zeros for the added reference area
                 # Ensure it has the same device and dtype as the edit_mask
                 ref_mask = torch.zeros(
                     [1, processed_ref_image.shape[1], processed_ref_image.shape[2]], # Shape: [1, H, W_ref]
                     device=edit_mask.device,
                     dtype=edit_mask.dtype
                 )
                 # Concatenate the masks
                 edit_mask = torch.cat([ref_mask, edit_mask], dim=-1)

                 # *** CRITICAL: Set slice_w to the actual final width of the added reference image ***
                 slice_w = processed_ref_image.shape[-1]

        H, W = edit_image.shape[-2:] # Get dimensions AFTER potential concatenation
        print(f"--- Before Final Scaling ---")
        print(f"Image dimensions (H, W): {H}, {W}")
        print(f"Current slice_w: {slice_w}")

        scale = min(1.0, math.sqrt(self.max_seq_len * 2 / ((H / self.d) * (W / self.d))))
        print(f"Calculated scale: {scale}")

        rH = int(H * scale) // self.d * self.d  # ensure divisible by self.d
        rW = int(W * scale) // self.d * self.d
        slice_w = math.ceil(slice_w * scale)

        print(f"Target dimensions (rH, rW) after scaling and grid alignment: {rH}, {rW}")
        print(f"Scaled slice_w after scaling and grid alignment: {slice_w}")
        print(f"--- End of Scaling Info ---")

        edit_image = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_image)
        edit_mask = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_mask)
        content_image = edit_image
        if use_change:
            change_image = edit_image * edit_mask
            edit_image = edit_image * (1 - edit_mask)
        else:
            change_image = None
        return edit_image, edit_mask, change_image, content_image, out_h, out_w, slice_w


    def postprocess(self, image, slice_w, out_w, out_h):
        w, h = image.size
        if slice_w > 0:
            output_image = image.crop((slice_w, 0, w, h))
            output_image = output_image.resize((out_w, out_h))
        else:
            output_image = image
        return output_image