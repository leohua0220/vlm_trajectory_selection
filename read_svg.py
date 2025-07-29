from lxml import etree


def parse_svg_with_lxml(file_path):
    """
    Parses an SVG file and prints the root element tag and attributes.
    """
    try:
        # Use the XML parser from lxml to parse the SVG file
        tree = etree.parse(file_path)
        root = tree.getroot()

        # Print the root tag and its attributes
        print(f"Root element tag: {root.tag}")
        print("Root element attributes:")
        for key, value in root.attrib.items():
            print(f"  {key}: {value}")

        # You can iterate through all elements in the SVG tree
        print("\nIterating through some child elements:")
        for element in root.iter():
            print(f"  Tag: {element.tag}, Attributes: {element.attrib}")

    except etree.XMLSyntaxError as e:
        print(f"Error parsing the SVG file: {e}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")

# Example usage:

parse_svg_with_lxml('/home/hua/Desktop/Frenetix-Motion-Planner/logs/ZAM_Tjunction-1_42_T-1/plots/ZAM_Tjunction-1_42_T-1_0.svg')