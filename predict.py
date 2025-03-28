# predict.py
import argparse
import glob
import importlib
import io
import os
import sys
import subprocess
import shlex



subprocess.run(shlex.split('pip install scepter --no-deps'))
subprocess.run(shlex.split('pip install numpy==1.26'))
subprocess.run(shlex.split('pip install flash-attn --no-build-isolation'),
               env=os.environ | {'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"})

from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.distribute import we
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

from cog import Input, Path, BasePredictor
from typing import List

from inference.registry import INFERENCES

fs_list = [ # keep this as class variable if needed across predictions, otherwise move to setup
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False), # Using ./cache for temp dir
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)


os.environ["FLUX_FILL_PATH"] = "hf://black-forest-labs/FLUX.1-Fill-dev"
# os.environ["PORTRAIT_MODEL_PATH"] = "hf://ali-vilab/ACE_Plus@portrait/comfyui_portrait_lora64.safetensors"
# os.environ["SUBJECT_MODEL_PATH"] = "hf://ali-vilab/ACE_Plus@subject/comfyui_subject_lora16.safetensors"
# os.environ["LOCAL_MODEL_PATH"] = "hf://ali-vilab/ACE_Plus@local_editing/comfyui_local_lora16.safetensors"
os.environ["ACE_PLUS_FFT_MODEL"] = "hf://ali-vilab/ACE_Plus@ace_plus_fft.safetensors"

os.environ["FLUX_FILL_PATH"]=FS.get_dir_to_local_dir(os.environ["FLUX_FILL_PATH"])
# os.environ["PORTRAIT_MODEL_PATH"]=FS.get_from(os.environ["PORTRAIT_MODEL_PATH"])
# os.environ["SUBJECT_MODEL_PATH"]=FS.get_from(os.environ["SUBJECT_MODEL_PATH"])
# os.environ["LOCAL_MODEL_PATH"]=FS.get_from(os.environ["LOCAL_MODEL_PATH"])
os.environ["ACE_PLUS_FFT_MODEL"]=FS.get_from(os.environ["ACE_PLUS_FFT_MODEL"])

class Predictor(BasePredictor):
    pipe = None # Initialize pipe as None at class level

    def setup(self):

        self.model_yamls = ["./config/ace_plus_fft.yaml"]
        self.model_choices = dict()
        self.default_model_name = ''
        self.edit_type_dict = {}
        self.edit_type_list = []
        self.default_type_list = []
        for i in self.model_yamls:
            model_cfg = Config(load=True, cfg_file=i)
            model_name = model_cfg.VERSION
            if model_cfg.IS_DEFAULT: self.default_model_name = model_name
            self.model_choices[model_name] = model_cfg
            for preprocessor in model_cfg.get("PREPROCESSOR", []):
                if preprocessor["TYPE"] in self.edit_type_dict:
                    continue
                self.edit_type_dict[preprocessor["TYPE"]] = preprocessor
                self.default_type_list.append(preprocessor["TYPE"])
        print('Models: ', self.model_choices.keys())
        assert len(self.model_choices) > 0
        if self.default_model_name == "": self.default_model_name = list(self.model_choices.keys())[0]
        self.model_name = self.default_model_name
        pipe_cfg = self.model_choices[self.default_model_name]
        self.pipe = INFERENCES.build(pipe_cfg)

        # self.lora_path = {
        #     "fft" : os.environ["ACE_PLUS_FFT_MODEL"],
        #     "portrait" : os.environ["PORTRAIT_MODEL_PATH"],
        #     "subject" : os.environ["SUBJECT_MODEL_PATH"],
        #     "edit" : os.environ["LOCAL_MODEL_PATH"],
        # }

    def predict(
        self,
        instruction: str = Input(description="The instruction for editing or generating!", default=""),
        lora: str = Input(description="The task/lora type", default='fft', choices=['fft', 'portrait', 'subject', 'edit']),
        output_w: int = Input(description="The width of output image", default=1440),
        output_h: int = Input(description="The height of output image", default=1440),
        input_image: Path = Input(description="The input image (optional)", default=None),
        input_mask: Path = Input(description="The input mask (optional)", default=None),
        input_reference_image: Path = Input(description="The input reference image (optional)", default=None),
        seed: int = Input(description="The seed for generation (default: -1 for random)", default=-1),
        sample_steps: int = Input(description="The sample step for generation (optional, default: 50)", default=50),
        guide_scale: float = Input(description="The guide scale for generation (optional, default: 50)", default=50),
        repainting_scale: float = Input(description="The repainting scale for content filling generation (optional, default: 0.0)", default=0.0),
        use_change: bool = Input(description="Use change (optional, default: True)", default=True),
        keep_pixels: bool = Input(description="Keep pixels (optional, default: True)", default=True),
        keep_pixels_rate: float = Input(description="Keep pixels rate (optional, default: 0.8)", default=0.8),

    ) -> List[Path]:
        pre_edit_image = None
        pre_edit_mask = None
        pre_ref_image = None

        if input_image:
            pre_edit_image = Image.open(str(input_image)).convert("RGB") # Open Path as string and convert to RGB
        if input_mask:
            pre_edit_mask = Image.open(str(input_mask)).convert("L") # Open Path as string and convert to L
        if input_reference_image:
            pre_ref_image = Image.open(str(input_reference_image)).convert("RGB") # Open Path as string and convert to RGB

        image, _, _, _, used_seed = self.pipe( # Use class-level Predictor.pipe
            reference_image=pre_ref_image,
            edit_image=pre_edit_image,
            edit_mask=pre_edit_mask,
            prompt=instruction,
            output_height=output_h,
            output_width=output_w,
            sampler='flow_euler',
            sample_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed,
            repainting_scale=repainting_scale,
            use_change=use_change,
            keep_pixels=keep_pixels,
            keep_pixels_rate=keep_pixels_rate
        )

        output_path = "output.png" # Fixed output path
        image.save(output_path) # Save directly to the output_path
        return [Path(output_path)] # Return output path as Cog Path