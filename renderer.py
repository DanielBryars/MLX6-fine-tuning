from PIL import Image, ImageDraw
import re

START_DIAGRAM = "<StartDiagram>"
END_DIAGRAM = '<EndDiagram>'

SMALL = "<Small>"
MEDIUM = "<Medium>"
LARGE = "<Large>"

TOP_LEFT = "<TopLeft>"
TOP_RIGHT = "<TopRight>"
BOTTOM_LEFT = "<BottomLeft>"
BOTTOM_RIGHT = "<BottomRight>"

CIRCLE = "<Circle>"
SQUARE = "<Square>"
TRIANGLE = "<Triangle>"

SIZES = {
    SMALL: 20,
    MEDIUM: 40,
    LARGE: 60
}

POSITIONS = {
    TOP_LEFT: (50, 50),
    TOP_RIGHT: (200, 50),
    BOTTOM_LEFT: (50, 200),
    BOTTOM_RIGHT: (200, 200)
}

def circle(draw, pos, size):
    x, y = pos
    r = size // 2
    draw.ellipse((x - r, y - r, x + r, y + r), outline="black", fill="lightblue")

def square(draw, pos, size):
    x, y = pos
    r = size // 2
    draw.rectangle((x - r, y - r, x + r, y + r), outline="black", fill="lightgreen")

def triangle(draw, pos, size):
    x, y = pos
    h = size
    points = [(x, y - h//2), (x - h//2, y + h//2), (x + h//2, y + h//2)]
    draw.polygon(points, outline="black", fill="lightpink")

SHAPES = {
    CIRCLE: circle,
    SQUARE: square,
    TRIANGLE: triangle
}

def render(dsl_string, image_size=(250, 250)):
    tokens = re.findall(r"<[^>]+>", dsl_string)

    if tokens[0] != START_DIAGRAM or tokens[-1] != END_DIAGRAM:
        raise ValueError(f"Diagram must start with {START_DIAGRAM} and end with {END_DIAGRAM}")

    tokens = tokens[1:-1]  # remove <StartDiagram> and <EndDiagram>

    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    i = 0
    #SIZE SHAPE POSITION
    #<Large><Circle><TopLeft> 
    while i < len(tokens):
        size = SIZES.get(tokens[i])
        if size is None:
            raise ValueError(f"Expected size token, got {tokens[i]}")
        
        shape = tokens[i + 1] 
        if shape not in SHAPES:
            raise ValueError(f"Unknown shape: {shape}")
        render_method = SHAPES[shape]
        
        pos_token = tokens[i + 2]
        pos = POSITIONS.get(pos_token)
        if pos is None:
            raise ValueError(f"Unknown position: {pos_token}")
        
        render_method(draw, pos, size)

        i += 3

    return img

if __name__ == "__main__":

    # Example usage
    dsl = '''<StartDiagram> 
    <Large><Square><TopLeft> 
    <Large><Square><TopRight> 
    <Large><Square><BottomLeft> 
    <Large><Square><BottomRight> 
    <EndDiagram>'''
    img = render(dsl)
    img.show()

    
    #dsl = "<StartDiagram> <Large><Circle><TopLeft> <Small> <Triangle> <BottomRight><EndDiagram>"
    #img = render(dsl)
    #img.show()#
