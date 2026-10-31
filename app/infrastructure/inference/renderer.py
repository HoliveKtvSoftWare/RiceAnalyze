"""PIL rendering of LabelMe polygon shapes."""

from PIL import Image, ImageDraw

from .labelme import prepare_preview_points


def draw_polygons_on_image(image, shapes, colors=None, draw_first=("out",),
                           outline_labels=(), outline_width=None,
                           preview_smooth=False, smooth_method="chaikin", smooth_iterations=2):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if colors is None:
        colors = {"out": (255, 0, 0, 64), "in": (0, 255, 0, 64),
                  "small": (0, 0, 255, 64), "big": (255, 255, 0, 64)}
    draw_first = tuple(draw_first or ())
    outline_labels = tuple(outline_labels or ())
    default_color = (255, 255, 0, 64)
    img_w, img_h = image.size
    if outline_width is None:
        outline_width = max(2, int(round(min(img_w, img_h) / 350)))

    def normalize_points(points):
        normalized = []
        for point in points:
            try:
                x = max(0, min(int(round(float(point[0]))), img_w - 1))
                y = max(0, min(int(round(float(point[1]))), img_h - 1))
                normalized.append((x, y))
            except (TypeError, ValueError, IndexError):
                continue
        return normalized

    def draw_one(shape):
        source_points = shape["points"]
        if preview_smooth:
            source_points = prepare_preview_points(source_points, method=smooth_method,
                                                   iterations=smooth_iterations)
        points = normalize_points(source_points)
        if len(points) < 3:
            return
        color = colors.get(shape["label"], default_color)
        if shape["label"] in outline_labels:
            draw.line(list(points) + [points[0]], fill=color, width=outline_width, joint="curve")
        else:
            draw.polygon(points, fill=color)

    for shape in shapes:
        if shape["label"] in draw_first:
            draw_one(shape)
    for shape in shapes:
        if shape["label"] not in draw_first:
            draw_one(shape)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
