"""

Boxplots of Elo ratings with 95% confidence intervals for each method.

Invocation:
    python draw_boxplots.py results.txt boxplots.png

@kylel

"""

import hashlib
import re
from pathlib import Path

import click
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import requests

# AI2 Colors
AI2_PINK = "#f0529c"
AI2_DARK_TEAL = "#0a3235"
AI2_TEAL = "#105257"

# Name mappings
NAME_DISPLAY_MAP = {"pdelf": "olmOCR", "mineru": "MinerU", "marker": "Marker", "gotocr_format": "GOTOCR"}


def download_and_cache_file(url, cache_dir=None):
    """Download a file and cache it locally."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "elo_plot"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from URL hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    file_name = url.split("/")[-1]
    cached_path = cache_dir / f"{url_hash}_{file_name}"

    if not cached_path.exists():
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(cached_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    return str(cached_path)


def parse_elo_data(file_path):
    """Parse Elo ratings data from a text file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Regular expression to match the data lines
    pattern = r"(\w+)\s+(\d+\.\d+)\s*±\s*(\d+\.\d+)\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]"
    matches = re.finditer(pattern, content)

    # Initialize lists to store data
    names = []
    medians = []
    errors = []
    ci_low = []
    ci_high = []

    for match in matches:
        names.append(match.group(1))
        medians.append(float(match.group(2)))
        errors.append(float(match.group(3)))
        ci_low.append(float(match.group(4)))
        ci_high.append(float(match.group(5)))

    return names, medians, errors, ci_low, ci_high


def create_boxplot(names, medians, errors, ci_low, ci_high, output_path, font_path):
    """Create and save a boxplot of Elo ratings."""
    # Set up Manrope font
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Manrope"
    plt.rcParams["font.weight"] = "medium"

    # Define colors - pdelf in pink, others in shades of teal/grey based on performance
    max_median = max(medians)
    colors = []
    for i, median in enumerate(medians):
        if names[i] == "pdelf":
            colors.append(AI2_PINK)
        else:
            # Calculate a shade between dark teal and grey based on performance
            performance_ratio = (median - min(medians)) / (max_median - min(medians))
            base_color = np.array(tuple(int(AI2_DARK_TEAL[i : i + 2], 16) for i in (1, 3, 5))) / 255.0
            grey = np.array([0.7, 0.7, 0.7])  # Light grey
            color = tuple(np.clip(base_color * performance_ratio + grey * (1 - performance_ratio), 0, 1))
            colors.append(color)

    # Create box plot data
    box_data = []
    for i in range(len(names)):
        q1 = medians[i] - errors[i]
        q3 = medians[i] + errors[i]
        box_data.append([ci_low[i], q1, medians[i], q3, ci_high[i]])

    # Create box plot with smaller width and spacing
    plt.figure(figsize=(4, 4))
    bp = plt.boxplot(
        box_data,
        labels=[NAME_DISPLAY_MAP[name] for name in names],
        whis=1.5,
        patch_artist=True,
        widths=0.15,  # Make boxes much narrower
        medianprops=dict(color="black"),  # Make median line black
        positions=np.arange(len(names)) * 0.25,
    )  # Reduce spacing between boxes significantly

    # Color each box
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Style the plot
    # plt.ylabel("Elo Rating", fontsize=12, color=AI2_DARK_TEAL)
    plt.xticks(
        np.arange(len(names)) * 0.25,  # Match positions from boxplot
        [NAME_DISPLAY_MAP[name] for name in names],
        rotation=45,
        ha="right",
        color=AI2_DARK_TEAL,
    )
    plt.yticks(color=AI2_DARK_TEAL)

    # Set x-axis limits to maintain proper spacing
    plt.xlim(-0.1, (len(names) - 1) * 0.25 + 0.1)

    # Remove the title and adjust the layout
    plt.tight_layout()

    # Remove spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Add left spine only
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["left"].set_color(AI2_DARK_TEAL)
    plt.gca().spines["left"].set_linewidth(0.5)

    # Add bottom spine only
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["bottom"].set_color(AI2_DARK_TEAL)
    plt.gca().spines["bottom"].set_linewidth(0.5)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--manrope-medium-font-path",
    type=str,
    default="https://dolma-artifacts.org/Manrope-Medium.ttf",
    help="Path to the Manrope Medium font file (local path or URL)",
)
def main(input_file, output_file, manrope_medium_font_path):
    """Generate a boxplot from Elo ratings data.

    INPUT_FILE: Path to the text file containing Elo ratings data
    OUTPUT_FILE: Path where the plot should be saved
    """
    try:
        # Handle font path - download and cache if it's a URL
        if manrope_medium_font_path.startswith(("http://", "https://")):
            font_path = download_and_cache_file(manrope_medium_font_path)
        else:
            font_path = manrope_medium_font_path

        # Parse the data
        names, medians, errors, ci_low, ci_high = parse_elo_data(input_file)

        # Create and save the plot
        create_boxplot(names, medians, errors, ci_low, ci_high, output_file, font_path)
        click.echo(f"Plot successfully saved to {output_file}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
