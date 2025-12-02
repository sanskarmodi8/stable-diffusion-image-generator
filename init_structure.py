"""
Project Structure Generator for Stable Diffusion Image Generator.

Creates all required directories and placeholder files with compliant
docstrings. Safe for repeated runs — will not overwrite existing files.
"""

import os

# Placeholder docstrings
MODULE_PLACEHOLDER = '"""Auto-generated placeholder module for Stable Diffusion Image Generator."""\n'
INIT_PLACEHOLDER = '"""Package initialization file for Stable Diffusion Image Generator."""\n'


# Utility functions
def create_dir(path: str):
    """Create directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def create_file(path: str, content: str = MODULE_PLACEHOLDER):
    """Create a file only if it does not already exist."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)


def init_file(path: str):
    """Create an __init__.py with a placeholder docstring."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(INIT_PLACEHOLDER)


# Project Directory Structure

directories = [
    "app",
    "app/core",
    "app/models",
    "app/utils",
    "app/presets",
    "app/upscaler",
    "assets",
    "assets/samples",
    "assets/lora",
]


# File Definitions

files = {
    # Entry
    "main.py": MODULE_PLACEHOLDER,

    # Core pipeline + generation modules
    "app/pipeline.py": MODULE_PLACEHOLDER,
    "app/generator.py": MODULE_PLACEHOLDER,
    "app/img2img.py": MODULE_PLACEHOLDER,

    # UI
    "app/ui.py": MODULE_PLACEHOLDER,

    # Presets
    "app/presets/styles.py": MODULE_PLACEHOLDER,

    # Upscaler
    "app/upscaler/realesrgan.py": MODULE_PLACEHOLDER,

    # Utils
    "app/utils/history.py": MODULE_PLACEHOLDER,
    "app/utils/seed.py": MODULE_PLACEHOLDER,
    "app/utils/logger.py": MODULE_PLACEHOLDER,

    # Models or reference files
    "app/models/metadata.py": MODULE_PLACEHOLDER,

    # Root files
    "requirements.txt": MODULE_PLACEHOLDER,
    "README.md": MODULE_PLACEHOLDER,
    "LICENSE": MODULE_PLACEHOLDER,
}


# Build the structure

for d in directories:
    create_dir(d)
    init_file(os.path.join(d, "__init__.py"))

for path, content in files.items():
    create_file(path, content)

print("Stable Diffusion Image Generator project structure created successfully!")
