import subprocess
import os
from pathlib import Path

def render_drawio_to_png(input_path, output_path, drawio_path="drawio"):
    """
    Renders a draw.io file to a PNG image using the draw.io CLI.

    Args:
        input_path (str): Path to the .drawio or .drawio.xml file.
        output_path (str): Desired path for the output PNG file.
        drawio_path (str): Path to the drawio CLI (default assumes it's in PATH).
    """
    script_dir = Path(__file__).resolve().parent

    input_path = script_dir / input_path

    output_path = script_dir / output_path


    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    cmd = [
        drawio_path,
        "--export",
        "--format", "png",
        "--output", output_path,
        input_path
    ]

    subprocess.run(cmd, check=True)
    print(f"Saved image to {output_path}")

# Example usage
render_drawio_to_png("shapes.drawio", "shapes.png")
