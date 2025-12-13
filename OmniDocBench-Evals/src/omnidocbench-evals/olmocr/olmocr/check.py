import importlib.util
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def check_poppler_version():
    try:
        result = subprocess.run(["pdftoppm", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stderr.startswith("pdftoppm"):
            logger.info("pdftoppm is installed and working.")
        else:
            logger.error("pdftoppm is installed but returned an error.")
            sys.exit(1)
    except FileNotFoundError:
        logger.error("pdftoppm is not installed.")
        logger.error("Check the README in the https://github.com/allenai/olmocr/blob/main/README.md for installation instructions")
        sys.exit(1)


def check_sglang_version():
    if importlib.util.find_spec("sglang") is None:
        logger.error("Please make sure sglang is installed according to the latest instructions here: https://docs.sglang.ai/start/install.html")
        logger.error("Sglang needs to be installed with a separate command in order to find all dependencies properly.")
        sys.exit(1)


def check_torch_gpu_available(min_gpu_memory: int = 15 * 1024**3):
    try:
        import torch
    except:
        logger.error("Pytorch must be installed, visit https://pytorch.org/ for installation instructions")
        raise

    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        assert gpu_memory >= min_gpu_memory
    except:
        logger.error(f"Torch was not able to find a GPU with at least {min_gpu_memory // (1024 ** 3)} GB of RAM.")
        raise


if __name__ == "__main__":
    check_poppler_version()
    check_sglang_version()
