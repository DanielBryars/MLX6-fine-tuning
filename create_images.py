from renderer import *
import random

NEW_LINE = "\n"

def create_random_image_dsl(pretty=False):

    tokens = []
    tokens.append(START_DIAGRAM)
    if (pretty):
        tokens.append(NEW_LINE)

    for position in list(POSITIONS):
        nShapes = len(SHAPES)    
        iShape = random.randint(0,nShapes) 
        if (iShape == nShapes):
            #Empty skip this position
            continue
        
        shape = list(SHAPES)[iShape]

        nSizes = len(SIZES)
        iSize = random.randint(0,nSizes-1)
        size = list(SIZES)[iSize]

        #SIZE SHAPE POSITION
        #<Large><Circle><TopLeft> 
        tokens.append(size)
        tokens.append(shape)
        tokens.append(position)
        if (pretty):
            tokens.append(NEW_LINE)
        #print(x)

    tokens.append(END_DIAGRAM)

    dsl = "".join(tokens)
    return dsl

def create_random_image():
    dsl = create_random_image_dsl()
    image = render(dsl)
    return image


if __name__ == "__main__":
    dsl = create_random_image_dsl(pretty=True)
    print(dsl)
    image = render(dsl)
    image.show()

    a = input("s to save reference.png")
    if (a == "s"):
        print("Saving .....")
        image.save("reference.png", format="PNG")
        with open("reference.dsl.txt", "w") as f:
            f.write(dsl)
    else:
        print("Skipped saving")
