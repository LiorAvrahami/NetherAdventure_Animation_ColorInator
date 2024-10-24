import os
import pickle
from typing import Any, Optional
from attr import dataclass
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
from skin_frame_maps import *


@dataclass
class ActionSummeryForUndo():
    is_action_on_override_points: bool
    frame_xy: tuple[int, int]
    skin_xy_before_action: Optional[tuple[int, int]]


# -- Globals
# gui globals
b_need_redraw = False

# keys
is_ctrl_down = False

# cursors
frame_cursor_position: Optional[tuple[int, int]] = None
skin_cursor_position: Optional[tuple[int, int]] = None
frame_cursor_object = None
skin_cursor_object = None
skin_anchor_indicator_object = None

# Matplotlib Objects:
# the points on the screen that indicate where anchors were placed
frame_anchor_indicator_objects: Any = None
frame_override_indicator_objects: Any = None

skin_image_object: Any = None
frame_outline_image_object: Any = None
frame_image_object: Any = None
debug_skin_result_image_object: Any = None
result_image_object: Any = None


panels_axes = []
current_body_part_index = 0
current_skin_index = 1
debug_skin_index = 0
# transformation globals
# the recorded mapping for each body part. these points are interpolated to construct the complete mapping
# mapping_anchor_points[body_part_index][x_frame,y_frame] = (x_skin,y_skin)
mapping_anchor_points: list[dict[tuple[int, int], tuple[int, int]]] = []
# these points are not given to the interpolation, but applied after the interpolation in order to allow fine adjustments.
# mapping_override_points[body_part_index][x_frame,y_frame] = (x_skin,y_skin)
mapping_override_points: list[dict[tuple[int, int], tuple[int, int]]] = []
# point_submissions_for_undo[body_part_index] = list of is_anchor_point
point_submissions_for_undo: list[list[ActionSummeryForUndo]] = []

#
all_loaded_skins: list[np.ndarray]
all_loaded_body_parts: list[np.ndarray]
body_part_masks: list[np.ndarray]

# cached_colored_body_part_images[skin_index,body_part_index] = colored_image_of_body_part_with_given_skin
cached_colored_body_part_images: dict[tuple[int, int], np.ndarray] = {}

body_parts_folder: str = None


def start_gui_session():
    global frame_cursor_position, skin_selected_position, skin_cursor_position, \
        skin_image_object, frame_outline_image_object, frame_image_object, debug_skin_result_image_object, \
        result_image_object, frame_anchor_indicator_objects, frame_override_indicator_objects, \
        mapping_anchor_points, mapping_override_points, point_submissions_for_undo, \
        panels_axes, current_body_part_index, current_skin_index, \
        all_loaded_skins, all_loaded_body_parts

    mapping_anchor_points = [{} for i in range(len(all_loaded_body_parts))]
    mapping_override_points = [{} for i in range(len(all_loaded_body_parts))]
    point_submissions_for_undo = [[] for i in range(len(all_loaded_body_parts))]
    # Set a large figure size
    fig, axs = plt.subplots(2, 2, figsize=(11, 9.8))
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    panels_axes = axs.ravel()

    # Remove margins and spaces between the plots
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    frame_shape = get_frame_shape()

    # Store references to the image objects (AxesImage) created by imshow
    # Plot images - top left
    draw_checkerboard_background(panels_axes[0], all_loaded_skins[0].shape)
    skin_image_object = panels_axes[0].imshow(np.zeros(all_loaded_skins[0].shape))
    # Plot images - top right
    draw_checkerboard_background(panels_axes[1], frame_shape)
    frame_outline_image_object = panels_axes[1].imshow(all_loaded_body_parts[current_body_part_index])
    frame_image_object = panels_axes[1].imshow(np.zeros(frame_shape))
    # Plot images - bottom left
    draw_checkerboard_background(panels_axes[2], frame_shape)
    debug_skin_result_image_object = panels_axes[2].imshow(np.zeros(frame_shape))
    # Plot images - bottom right
    draw_checkerboard_background(panels_axes[3], frame_shape)
    result_image_object = panels_axes[3].imshow(np.zeros(frame_shape))

    frame_anchor_indicator_objects = panels_axes[1].plot([], [], "o", markersize=2, markeredgecolor="k",
                                                         markerfacecolor="white", fillstyle="full", markeredgewidth=0.8)[0]
    frame_override_indicator_objects = panels_axes[1].plot([], [], "o", markersize=2, markeredgecolor="r",
                                                           markerfacecolor="white", fillstyle="full", markeredgewidth=0.8)[0]

    update_panels_images()

    # Connect the click event to each subplot
    for i, ax in enumerate(panels_axes):
        fig.canvas.mpl_connect('button_press_event', lambda event, subplot_index=i: on_click(event, subplot_index=subplot_index))

    # Connect the keyboard event to the figure
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event))

    fig.canvas.mpl_connect('key_release_event', lambda event: on_key_release(event))

    plt.show()


def draw_checkerboard_background(axis, shape):
    c1 = [0, 0, 0, 100]
    c2 = [100, 100, 100, 100]
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    bg = np.full(shape, c1)
    for i in range(4):
        bg[:, :, i] = ((x + y) % 2) * c2[i]
    axis.imshow(bg)


def body_part_map_changed(body_part_index):
    for key in list(cached_colored_body_part_images.keys()):
        if key[1] == body_part_index:
            del cached_colored_body_part_images[key]


def update_body_part_images_cache(skin_index, body_part_index):
    frame_shape = get_frame_shape()
    full_skin_map = interpolate_map_for_body_part(
        mapping_anchor_points[body_part_index], mapping_override_points[body_part_index], frame_shape)
    colored_body_part = apply_skin_map_to_body_part(full_skin_map, all_loaded_skins[skin_index])
    colored_body_part[np.logical_not(body_part_masks[body_part_index]), :] = 0
    cached_colored_body_part_images[skin_index, body_part_index] = colored_body_part


def get_colored_body_part_image(skin_index, body_part_index):
    if (skin_index, body_part_index) not in cached_colored_body_part_images:
        update_body_part_images_cache(skin_index, body_part_index)
    return cached_colored_body_part_images[skin_index, body_part_index]


def get_complete_colored_picture(skin_index):
    colored_body_part_images = [get_colored_body_part_image(skin_index, body_part_index)
                                for body_part_index in range(len(all_loaded_body_parts))]
    out_image = overlay_images(colored_body_part_images)
    return out_image


def get_complete_mapping():
    frame_shape = get_frame_shape()
    body_part_maps = []
    for body_part_index in range(len(all_loaded_body_parts)):
        full_skin_map = np.zeros(frame_shape)
        full_skin_map[:, :, 3] = 255
        full_skin_map[:, :, :2] = interpolate_map_for_body_part(mapping_anchor_points[body_part_index],
                                                                mapping_override_points[body_part_index], frame_shape)
        nan_indexes = (full_skin_map[:, :, 0] == 0) * (full_skin_map[:, :, 1] == 0)
        full_skin_map[nan_indexes, 3] = 0
        full_skin_map[np.logical_not(body_part_masks[body_part_index]), :] = 0
        body_part_maps.append(full_skin_map)
    out_map = overlay_images(body_part_maps)
    return out_map


def update_panels_images():

    # top left panel
    skin_image_object.set_data(all_loaded_skins[current_skin_index])
    # top right panel
    frame_image_object.set_data(get_colored_body_part_image(current_skin_index, current_body_part_index))
    # bottom left panel
    result_image_object.set_data(get_complete_colored_picture(current_skin_index))
    # bottom right panel
    debug_skin_result_image_object.set_data(get_complete_colored_picture(debug_skin_index))

    update_cursor_positions()

    global b_need_redraw
    b_need_redraw = True


def update_cursor_positions():
    global frame_cursor_object, skin_cursor_object, skin_anchor_indicator_object

    frame_cursor_object = \
        update_single_cursor(frame_cursor_position, frame_cursor_object, panels_axes_index=1, color="g")
    skin_cursor_object = \
        update_single_cursor(skin_cursor_position, skin_cursor_object, panels_axes_index=0, color="g")

    skin_anchor_position = None
    if frame_cursor_position is not None:
        skin_anchor_position = \
            mapping_anchor_points[current_body_part_index][frame_cursor_position]\
            if frame_cursor_position in mapping_anchor_points[current_body_part_index] \
            else None
    skin_anchor_indicator_object = \
        update_single_cursor(skin_anchor_position, skin_anchor_indicator_object, panels_axes_index=0, color="r")


def update_single_cursor(cursor_position, cursor_object, panels_axes_index, color):
    global b_need_redraw

    if cursor_position is not None:
        if cursor_object is None:
            cursor_object, = panels_axes[panels_axes_index].plot([cursor_position[1]], [cursor_position[0]], marker="X",
                                                                 markeredgecolor=color, markerfacecolor="k", fillstyle="full",
                                                                 markersize=11, markeredgewidth=1.5)
            b_need_redraw = True
        else:
            cursor_object.set_data([cursor_position[1]], [cursor_position[0]])
            b_need_redraw = True
    else:
        if cursor_object is not None:
            cursor_object.remove()
            cursor_object = None
    return cursor_object

def update_frame_override_indicator_objects():
    global frame_override_indicator_objects,b_need_redraw
    frame_override_indicator_objects.set_data([xy[1] for xy in mapping_override_points[current_body_part_index].keys()],
                                              [xy[0] for xy in mapping_override_points[current_body_part_index].keys()])
    b_need_redraw = True

def update_frame_anchor_indicator_objects():
    global frame_anchor_indicator_objects,b_need_redraw
    frame_anchor_indicator_objects.set_data([xy[1] for xy in mapping_anchor_points[current_body_part_index].keys()],
                                              [xy[0] for xy in mapping_anchor_points[current_body_part_index].keys()])
    b_need_redraw = True

def prev_body_part():
    global current_body_part_index
    if current_body_part_index == 0:
        return
    current_body_part_index -= 1
    frame_outline_image_object.set_data(all_loaded_body_parts[current_body_part_index])
    update_panels_images()


def next_body_part():
    global current_body_part_index
    if current_body_part_index == len(all_loaded_body_parts) - 1:
        return
    current_body_part_index += 1
    frame_outline_image_object.set_data(all_loaded_body_parts[current_body_part_index])
    update_panels_images()


def prev_skin():
    global current_skin_index
    if current_skin_index == 0:
        return
    current_skin_index -= 1
    update_panels_images()


def next_skin():
    global current_skin_index
    if current_skin_index == len(all_loaded_skins) - 1:
        return
    current_skin_index += 1
    update_panels_images()


def undo_one_action():
    if len(point_submissions_for_undo[current_body_part_index]) == 0:
        return
    last_action_summery = point_submissions_for_undo[current_body_part_index].pop()
    if last_action_summery.is_action_on_override_points:
        if last_action_summery.skin_xy_before_action is not None:
            mapping_override_points[current_body_part_index][last_action_summery.frame_xy] = last_action_summery.skin_xy_before_action
        else:
            del mapping_override_points[current_body_part_index][last_action_summery.frame_xy]
    else:
        if last_action_summery.skin_xy_before_action is not None:
            mapping_anchor_points[current_body_part_index][last_action_summery.frame_xy] = last_action_summery.skin_xy_before_action
        else:
            del mapping_anchor_points[current_body_part_index][last_action_summery.frame_xy]

    update_frame_anchor_indicator_objects()
    body_part_map_changed(current_body_part_index)
    update_panels_images()


def submit_mapping_point_and_decide_whether_is_override():
    if is_ctrl_down:
        submit_override_point()
    else:
        submit_anchor_point()


def submit_anchor_point():
    # point_submissions_for_undo[current_body_part_index].append(True)
    if skin_cursor_position is None or frame_cursor_position is None:
        return

    if frame_cursor_position in mapping_anchor_points[current_body_part_index]:
        skin_value_before_submission = mapping_anchor_points[current_body_part_index][frame_cursor_position]
    else:
        skin_value_before_submission = None
    point_submissions_for_undo[current_body_part_index].append(ActionSummeryForUndo(
        is_action_on_override_points=False,
        frame_xy=frame_cursor_position,
        skin_xy_before_action=skin_value_before_submission))

    mapping_anchor_points[current_body_part_index][frame_cursor_position] = skin_cursor_position
    update_frame_anchor_indicator_objects()
    body_part_map_changed(current_body_part_index)
    update_panels_images()


def submit_override_point():
    # point_submissions_for_undo[current_body_part_index].append(False)
    if skin_cursor_position is None or frame_cursor_position is None:
        return

    if frame_cursor_position in mapping_anchor_points[current_body_part_index]:
        skin_value_before_submission = mapping_anchor_points[current_body_part_index][frame_cursor_position]
    else:
        skin_value_before_submission = None
    point_submissions_for_undo[current_body_part_index].append(ActionSummeryForUndo(
        is_action_on_override_points=True,
        frame_xy=frame_cursor_position,
        skin_xy_before_action=skin_value_before_submission))

    mapping_override_points[current_body_part_index][frame_cursor_position] = skin_cursor_position
    update_frame_override_indicator_objects()
    frame_override_indicator_objects.set_data([xy[1] for xy in mapping_override_points[current_body_part_index].keys()],
                                            [xy[0] for xy in mapping_override_points[current_body_part_index].keys()])
    body_part_map_changed(current_body_part_index)
    update_panels_images()


def is_zoom_or_pan():
    return plt.gcf().canvas.toolbar.mode != ""


def get_frame_shape():
    return all_loaded_body_parts[0].shape


def on_click(event, subplot_index):
    global skin_cursor_position, frame_cursor_position
    if event.inaxes != panels_axes[subplot_index] or is_zoom_or_pan():
        return
    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
    if subplot_index == 0:
        # top left - skins
        skin_cursor_position = y, x
        submit_mapping_point_and_decide_whether_is_override()
    if subplot_index == 1:
        # top right - body part
        frame_cursor_position = y, x
        # if
    if subplot_index == 3:
        # set skin as debug skin
        global debug_skin_index
        debug_skin_index = current_skin_index
        update_panels_images()

    update_cursor_positions()

    draw_if_needed()


def on_key_press(event):
    global skin_cursor_position, frame_cursor_position, b_need_redraw, is_ctrl_down
    if event.key == 'ctrl+z':
        undo_one_action()
        pass
    if event.key == 'w':
        move_skin_cursor(0, -1)
    if event.key == 'a':
        move_skin_cursor(-1, 0)
    if event.key == 's':
        move_skin_cursor(0, 1)
    if event.key == 'd':
        move_skin_cursor(1, 0)
    if event.key == 'up':
        move_frame_cursor(0, -1)
    if event.key == 'left':
        move_frame_cursor(-1, 0)
    if event.key == 'down':
        move_frame_cursor(0, 1)
    if event.key == 'right':
        move_frame_cursor(1, 0)
    if event.key == 'space':
        submit_anchor_point()
    if event.key == 'ctrl+space':
        submit_override_point()
    if event.key == 'h':
        b_need_redraw = True
        frame_image_object.set_alpha(0.5)
    if event.key == ',':
        prev_body_part()
    if event.key == '.':
        next_body_part()
    if event.key == 'q':
        prev_skin()
    if event.key == 'e':
        next_skin()
    if event.key == 'p':
        if frame_anchor_indicator_objects.get_alpha() == 0:
            frame_anchor_indicator_objects.set_alpha(1.0)
            frame_override_indicator_objects.set_alpha(1.0)
        else:
            frame_anchor_indicator_objects.set_alpha(0)
            frame_override_indicator_objects.set_alpha(0)
        b_need_redraw = True
    if event.key == 'enter':
        complete_map = get_complete_mapping()
        export_to_png(complete_map)
        save_progress_to_pickle()
    if event.key == 'ctrl+l':
        load_progress_from_pickle()
    if event.key == 'control':
        is_ctrl_down = True
    draw_if_needed()


def on_key_release(event):
    if event.key == 'control':
        global is_ctrl_down
        is_ctrl_down = False
    if event.key == 'enter':
        pass
    if event.key == 'h':
        frame_image_object.set_alpha(1)

    draw_if_needed()


def move_skin_cursor(dx, dy):
    global skin_cursor_position
    if skin_cursor_position is None:
        return
    skin_cursor_position = (skin_cursor_position[0] + dy, skin_cursor_position[1] + dx)
    update_cursor_positions()


def move_frame_cursor(dx, dy):
    global frame_cursor_position
    if frame_cursor_position is None:
        return
    frame_cursor_position = (frame_cursor_position[0] + dy, frame_cursor_position[1] + dx)
    update_cursor_positions()


def draw_if_needed():
    if b_need_redraw:
        plt.draw()


# Function to open a file dialog and load images
def load_images():
    global body_parts_folder
    # Initialize Tkinter and hide the root window
    root = Tk()
    root.withdraw()

    # ask user for skins folder
    # filedialog.askdirectory(title="select skins folder") <-- this was commented out because static skins folder is better
    skins = r"skins"
    body_parts_folder = filedialog.askdirectory(title="select body parts folder")

    skins_images = []
    for file_path in os.listdir(skins):
        if os.path.splitext(file_path)[1] == ".pickle":
            continue
        file_path = os.path.join(skins, file_path)
        img = Image.open(file_path)
        skins_images.append(np.array(img))

    body_part_images: list[Any] = []
    body_part_masks: list[Any] = []
    for i, file_name in enumerate(sorted(os.listdir(body_parts_folder))):
        if file_name in ["SkinMapToolProgress.pickle", "SkinMap.png"]:
            continue
        file_path = os.path.join(body_parts_folder, file_name)
        img = Image.open(file_path)
        body_part_images.append(np.array(img))
        body_part_masks.append(body_part_images[i][:, :, 3] != 0)

    return skins_images, body_part_images, body_part_masks


def export_to_png(image_data):
    # Create a PIL image from the numpy array and save it as a transparent PNG
    pil_image = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    filepath = os.path.join(body_parts_folder, "SkinMap.png")
    pil_image.save(filepath, 'PNG')
    print(f"Image saved as {filepath}")

    os.system(f"explorer \"{os.path.dirname(filepath)}\"".replace("/",os.sep).replace("\\",os.sep))


def save_progress_to_pickle():
    filepath = os.path.join(body_parts_folder, "SkinMapToolProgress.pickle")
    data_to_save = {
        "mapping_anchor_points": mapping_anchor_points,
        "mapping_override_points": mapping_override_points,
        "point_submissions_for_undo": point_submissions_for_undo
    }
    # Save the data to a pickle file
    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"progress saved as {filepath}")


def load_progress_from_pickle():
    global mapping_anchor_points, mapping_override_points, point_submissions_for_undo, cached_colored_body_part_images
    filepath = os.path.join(body_parts_folder, "SkinMapToolProgress.pickle")
    # Load the data from the pickle file
    with open(filepath, "rb") as f:
        loaded_data = pickle.load(f)
    # Extract the data back into variables
    mapping_anchor_points = loaded_data["mapping_anchor_points"]
    mapping_override_points = loaded_data["mapping_override_points"]
    point_submissions_for_undo = loaded_data["point_submissions_for_undo"]
    cached_colored_body_part_images = {}
    update_frame_anchor_indicator_objects()
    update_frame_override_indicator_objects()
    update_panels_images()
    


if __name__ == "__main__":
    # Load images using file dialog
    all_loaded_skins, all_loaded_body_parts, body_part_masks = load_images()
    start_gui_session()
