import numpy as np
import torch
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

class ImageEditor:
    def __init__(self):
        self.sam_model = None
        self.mask_generator = None
        self.sd_pipeline = None
        self.src_img = None
        self.masks = None

    def load_models(self):
        # Load SAM model
        model_type = "vit_h"
        model_path = "sam_chkpts/sam_vit_h_4b8939.pth"
        self.sam_model = sam_model_registry[model_type](checkpoint=model_path).to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.99,
            stability_score_offset=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        # Load Stable Diffusion model
        sd_model = 'stabilityai/stable-diffusion-2-inpainting'
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_model, subfolder="scheduler")
        self.sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model, scheduler=scheduler, revision="fp16", safety_checker=None, torch_dtype=torch.float16
        ).to("cuda")

    def load_and_process_image(self, image_path, target_size=(512, 512)):
        self.src_img = Image.open(image_path)
        self.src_img = self.src_img.resize(target_size, Image.LANCZOS)
        return self.src_img

    def generate_masks(self):
        src_img_array = np.array(self.src_img)
        self.masks = self.mask_generator.generate(src_img_array)
        self.masks = [mask for mask in self.masks if mask['area'] >= 20000]
        return self.masks

    def get_mask_image(self, mask_id):
        segmentation_mask = self.masks[mask_id]['segmentation']
        return Image.fromarray(segmentation_mask)

    def inpaint_image(self, mask_id, prompt):
        mask_img = self.get_mask_image(mask_id)
        generator = torch.Generator().manual_seed(0)

        image_gen = self.sd_pipeline(
            prompt=prompt,
            image=self.src_img,
            mask_image=mask_img,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            output_type="pil",
        )

        return image_gen.images[0]

# Usage example:
# editor = ImageEditor()
# editor.load_models()
# editor.load_and_process_image("path_to_image.jpg")
# masks = editor.generate_masks()
# result = editor.inpaint_image(0, "sky with lightning and thunderstorms")