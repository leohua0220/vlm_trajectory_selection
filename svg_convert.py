import cairosvg


def convert_svg_to_png(svg_file_path, output_png_path, dpi=100):
    """
    Converts an SVG file to a PNG image with a specified DPI.

    Args:
        svg_file_path (str): Path to the input SVG file.
        output_png_path (str): Path to save the output PNG file.
        dpi (int): Dots per inch for the output image resolution.
    """
    try:
        # Convert the SVG file to PNG
        cairosvg.svg2png(url=svg_file_path, write_to=output_png_path, dpi=dpi)
        print(f"Successfully converted '{svg_file_path}' to '{output_png_path}'.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# Example usage:
convert_svg_to_png('plots/ZAM_Zip-1_51_T-1_0.svg',
                   'plots/ZAM_Zip-1_51_T-1_0.png', 100)