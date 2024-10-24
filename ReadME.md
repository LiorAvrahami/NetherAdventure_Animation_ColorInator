the left top panel is the skin, the top right panel is the frame.
the bottom left panel is the frame when the debug skin is applied.
the bottom right panel is the frame when the current skin is applied.

# Input

in present working directory there should be a folder named skins, this folder should contain some skins.
you will be prompted to supply a frame folder that contains the body parts of the frame.

# Ouput

press enter to export.
will save the final map and pickle file with the progress to the folder with the body parts.

# Usage:

click bottom right panel - set current skin to be debug skin
click frame and then skin - submit link between frame and skin.
ctrl + click - submit link between frame and skin. but this data point doesn't effect the interpolation

wasd - move the skin cursor
arrow keys - move the frame courser
space - submit link current frame courser pixel to skin cursor position
ctrl + space - submit, but this data point doesn't effect the interpolation

h - make coloring slightly transparent in body part panel

<,> - change body part
q,e - change skin

ctrl + z - undo

p - toggle show hide submitted mapping points

ctrl + l - load progress pickle file
enter - save map image and progress pickle file
