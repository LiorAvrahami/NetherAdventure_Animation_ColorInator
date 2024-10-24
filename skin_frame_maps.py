import numpy as np
from scipy.interpolate import LinearNDInterpolator


def interpolate_map_for_body_part(mapping_anchor_points: dict[tuple, tuple],
                                  mapping_override_points: dict[tuple, tuple],
                                  frame_shape):
    # mapping_anchor_points = [((x_frame,y_frame),(x_skin,y_skin)),((x_frame,y_frame),(x_skin,y_skin)),...]
    # mapping_override_points = [((x_frame,y_frame),(x_skin,y_skin)),((x_frame,y_frame),(x_skin,y_skin)),...]

    if len(mapping_anchor_points) >= 3:
        anchors_frame_xy = list(mapping_anchor_points.keys())
        anchors_skin_x = [mapping_anchor_points[frame_xy][0] for frame_xy in anchors_frame_xy]
        anchors_skin_y = [mapping_anchor_points[frame_xy][1] for frame_xy in anchors_frame_xy]

        frame_pixels_x,frame_pixels_y = np.meshgrid(np.arange(frame_shape[0]), np.arange(frame_shape[1]),indexing="ij") 

        full_skin_x_map = LinearNDInterpolator(anchors_frame_xy, anchors_skin_x)(frame_pixels_x, frame_pixels_y)
        full_skin_y_map = LinearNDInterpolator(anchors_frame_xy, anchors_skin_y)(frame_pixels_x, frame_pixels_y)

        full_skin_map = np.stack((full_skin_x_map, full_skin_y_map), axis=-1)
    else:
        full_skin_map = np.zeros((*frame_shape[:2], 2))

    for frame_xy in mapping_anchor_points:
        full_skin_map[frame_xy] = mapping_anchor_points[frame_xy]
    for frame_xy in mapping_override_points:
        full_skin_map[frame_xy] = mapping_override_points[frame_xy]
    full_skin_map[np.isnan(full_skin_map)] = 0
    return np.round(full_skin_map).astype(int)


def apply_skin_map_to_body_part(full_skin_map: np.ndarray, skin: np.ndarray):
    # full_skin_map.shape = (body_part_width,body_part_hight,2)
    full_skin_map_flat = full_skin_map[:, :, 0] * skin.shape[1] + full_skin_map[:, :, 1]
    colored_body_part = np.stack([skin[:, :, i].flatten()[full_skin_map_flat] for i in range(skin.shape[-1])], axis=-1)
    return colored_body_part

def overlay_images(images):
    """Overlays multiple images (assuming they all have the same dimensions)."""
    # Initialize with a fully transparent base (0 for RGB, 0 for Alpha)
    base_image = np.zeros_like(images[0], dtype=np.float32)

    for img in images:
        # Normalize the image (values between 0 and 1)
        img_normalized = img.astype(np.float32) / 255.0

        # Split RGBA channels
        r, g, b, alpha = img_normalized[..., 0], img_normalized[..., 1], img_normalized[..., 2], img_normalized[..., 3]
        
        # Composite the images by combining the current one with the base_image
        base_alpha = base_image[..., 3]
        new_alpha = alpha + base_alpha * (1 - alpha)
        
        base_image[..., 0] = (r * alpha + base_image[..., 0] * base_alpha * (1 - alpha)) / (new_alpha + 0.000001)
        base_image[..., 1] = (g * alpha + base_image[..., 1] * base_alpha * (1 - alpha)) / (new_alpha + 0.000001)
        base_image[..., 2] = (b * alpha + base_image[..., 2] * base_alpha * (1 - alpha)) / (new_alpha + 0.000001)
        base_image[..., 3] = new_alpha
    
    # Convert the base_image back to 0-255 range
    return np.clip(base_image * 255, 0, 255).astype(np.uint8)