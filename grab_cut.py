import os
import cv2
import numpy as np

# =========================
# Configuration
# =========================
INPUT_FOLDER = "data/owl_images"
OUTPUT_FOLDER = "data/owl_output"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

WINDOW_NAME = "Draw rectangle around owl"
MASK_WINDOW = "Mask"
FOREGROUND_WINDOW = "Foreground"

TARGET_SIZE = 256

# =========================
# Global state for GUI
# =========================
drawing = False
ix, iy = -1, -1
rect = None
rect_done = False
image = None
display = None


def reset_state(img):
    global drawing, ix, iy, rect, rect_done, image, display
    drawing = False
    ix, iy = -1, -1
    rect = None
    rect_done = False
    image = img
    display = img.copy()


def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, rect, rect_done, display, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = image.copy()
        cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
        display = temp

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = min(ix, x), min(iy, y)
        w, h = abs(x - ix), abs(y - iy)

        if w > 5 and h > 5:
            rect = (x0, y0, w, h)
            rect_done = True
            temp = image.copy()
            cv2.rectangle(temp, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
            display = temp


def run_grabcut(img, rect, iterations=5):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img,
        mask,
        rect,
        bgd_model,
        fgd_model,
        iterations,
        cv2.GC_INIT_WITH_RECT
    )

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    return binary_mask


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small gaps and connect nearby regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill internal holes
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)
    mask = cv2.bitwise_or(mask, holes)

    return mask


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def get_image_files(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTENSIONS:
            files.append(path)
    return files


def get_output_paths(input_path, output_folder):
    stem = os.path.splitext(os.path.basename(input_path))[0]

    image_dir = os.path.join(output_folder, "images_256")
    mask_dir = os.path.join(output_folder, "masks_256")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_out = os.path.join(image_dir, f"{stem}.png")
    mask_out = os.path.join(mask_dir, f"{stem}_mask.png")

    return image_out, mask_out


def crop_masked_square(img, mask, target_size=256, padding_ratio=0.1):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None, None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    side = max(w, h)

    # Add a little padding around the owl
    side = int(np.ceil(side * (1.0 + padding_ratio)))

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    img_h, img_w = img.shape[:2]

    # If crop goes outside image, pad image and mask
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        mask = cv2.copyMakeBorder(
            mask,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    resized_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(cropped_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    return resized_img, resized_mask


def process_single_image(image_path, output_folder, index, total):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[{index}/{total}] Could not load: {image_path}")
        return "error"
    # Resize to max dimension of 512 for better performance (preserve aspect ratio)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 512:
        scale = 512 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    reset_state(img)
    current_mask = None

    print("\n" + "=" * 60)
    print(f"[{index}/{total}] {os.path.basename(image_path)}")
    print("Instructions:")
    print("  - Drag a rectangle around the owl")
    print("  - Press 'g' to run GrabCut")
    print("  - Press 'r' to reset rectangle")
    print("  - Press 's' to save cropped 256x256 image and mask, then go next")
    print("  - Press 'n' to skip this image")
    print("  - Press 'q' or ESC to quit")

    while True:
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("g"):
            if not rect_done:
                print("Draw a rectangle first.")
                continue

            current_mask = run_grabcut(img, rect)
            current_mask = clean_mask(current_mask)
            current_mask = keep_largest_component(current_mask)

            foreground = cv2.bitwise_and(img, img, mask=current_mask)

            cv2.imshow(MASK_WINDOW, current_mask)
            cv2.imshow(FOREGROUND_WINDOW, foreground)
            print("GrabCut finished.")

        elif key == ord("r"):
            reset_state(img)
            current_mask = None
            cv2.destroyWindow(MASK_WINDOW)
            cv2.destroyWindow(FOREGROUND_WINDOW)
            print("Reset.")

        elif key == ord("s"):
            if current_mask is None:
                print("No mask yet. Press 'g' first.")
                continue

            cropped_img, cropped_mask = crop_masked_square(
                img,
                current_mask,
                target_size=TARGET_SIZE,
                padding_ratio=0.10
            )

            if cropped_img is None or cropped_mask is None:
                print("Mask appears empty. Could not crop.")
                continue

            image_out, mask_out = get_output_paths(image_path, output_folder)
            cv2.imwrite(image_out, cropped_img)
            cv2.imwrite(mask_out, cropped_mask)

            print(f"Saved cropped image: {image_out}")
            print(f"Saved cropped mask:  {mask_out}")

            cv2.destroyWindow(WINDOW_NAME)
            cv2.destroyWindow(MASK_WINDOW)
            cv2.destroyWindow(FOREGROUND_WINDOW)
            return "saved"

        elif key == ord("n"):
            print("Skipped.")
            cv2.destroyWindow(WINDOW_NAME)
            cv2.destroyWindow(MASK_WINDOW)
            cv2.destroyWindow(FOREGROUND_WINDOW)
            return "skipped"

        elif key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            return "quit"


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder does not exist: {INPUT_FOLDER}")
        return

    image_files = get_image_files(INPUT_FOLDER)
    if not image_files:
        print(f"No image files found in: {INPUT_FOLDER}")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, draw_rectangle)

    saved_count = 0
    skipped_count = 0

    for i, image_path in enumerate(image_files, start=1):
        result = process_single_image(image_path, OUTPUT_FOLDER, i, len(image_files))

        if result == "saved":
            saved_count += 1
        elif result == "skipped":
            skipped_count += 1
        elif result == "quit":
            break

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, draw_rectangle)

    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Saved:   {saved_count}")
    print(f"Skipped: {skipped_count}")


if __name__ == "__main__":
    main()