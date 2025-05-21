import xml.etree.ElementTree as ET

def create_drawio_diagram():
    # Create XML tree
    mxfile = ET.Element('mxfile', host="app.diagrams.net")
    diagram = ET.SubElement(mxfile, 'diagram', name="Shapes")

    # This line attaches the graph model correctly to the diagram
    mxGraphModel = ET.SubElement(diagram, 'mxGraphModel')
    root = ET.SubElement(mxGraphModel, 'root')

    # Required root cells
    ET.SubElement(root, 'mxCell', id="0")
    ET.SubElement(root, 'mxCell', id="1", parent="0")

    def add_shape(shape_id, shape_type, x, y, w, h, style):
        cell = ET.SubElement(root, 'mxCell', id=shape_id, value=shape_type, style=style,
                             vertex="1", parent="1")
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")

    # Square
    add_shape("2", "Square", 40, 40, 80, 80, "shape=rectangle;whiteSpace=wrap;html=1;")

    # Circle
    add_shape("3", "Circle", 160, 40, 80, 80, "shape=ellipse;whiteSpace=wrap;html=1;")

    # Triangle
    add_shape("4", "Triangle", 280, 40, 80, 80, "shape=triangle;whiteSpace=wrap;html=1;")

    return ET.ElementTree(mxfile)

# Save to file
tree = create_drawio_diagram()
tree.write("shapes.drawio", encoding="utf-8", xml_declaration=True)
