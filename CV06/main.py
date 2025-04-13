import math
import matplotlib.pyplot as plt


# Create L system string
def make_l_system(axiom, rules, how_many_times):
    result = axiom
    for i in range(how_many_times):
        new_result = ""
        for letter in result:
            if letter in rules:
                new_result += rules[letter]
            else:
                new_result += letter
        result = new_result
    return result


# Function to turn the L-system string into lines
def draw_l_system(commands, angle, start_x=0, start_y=0, start_angle=math.pi / 2):
    x = start_x
    y = start_y
    direction = start_angle
    lines = []
    saved_positions = []  # Stack for branching

    # Iterate through each command ins string
    for letter in commands:
        if letter == "F":  # Move forward and draw a line
            new_x = x + math.cos(direction)
            new_y = y + math.sin(direction)
            lines.append(((x, y), (new_x, new_y)))  # Add line to list
            x = new_x
            y = new_y
        elif letter == "b":  # Move forward without drawing
            x += math.cos(direction)
            y += math.sin(direction)
        elif letter == "+":  # Turn right
            direction -= angle
        elif letter == "-":  # Turn left
            direction += angle
        elif letter == "[":  # Save the current
            saved_positions.append((x, y, direction))
        elif letter == "]":  # Restore the last saved
            if len(saved_positions) > 0:
                x, y, direction = saved_positions.pop()

    return lines


# Function to show the drawing
def show_lines(lines):
    for line in lines:
        (x1, y1), (x2, y2) = line
        plt.plot([x1, x2], [y1, y2], color="black")

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.show()


# Parameters for each example
l_systems = [
    {
        "start": "F+F+F+F",
        "rules": {"F": "F+F-F-FF+F+F-F"},
        "angle": math.radians(90),
        "times": 3,
        "start_step": 100,
    },
    {
        "start": "F++F++F",
        "rules": {"F": "F+F--F+F"},
        "angle": math.radians(60),
        "times": 3,
        "start_step": 100,
    },
    {
        "start": "F",
        "rules": {"F": "F[+F]F[-F]F"},
        "angle": math.pi / 7,
        "times": 5,
        "start_step": 100,
    },
    {
        "start": "F",
        "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},
        "angle": math.pi / 8,
        "times": 3,
        "start_step": 100,
    },
]


if __name__ == "__main__":

    chosen = int(input("Choose an L-system (0-3): "))
    if not (chosen >= 0 and chosen <= 3):
        print("Invalid choice. Defaulting to 0.")
        chosen = 0

    # Prepare the system
    chosen = l_systems[chosen]
    rule = chosen["rules"]
    start = chosen["start"]
    angle = chosen["angle"]
    steps = chosen["times"]
    start_step = chosen["start_step"]

    # Make and draw the L-system
    pattern = make_l_system(start, rule, steps)
    print("Pattern length:", len(pattern))
    lines = draw_l_system(pattern, angle)
    show_lines(lines)
