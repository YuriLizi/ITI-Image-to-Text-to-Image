import os
import random
import shutil


def copy_random_images(input_folder, output_folder, images_per_folder=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 1
    for root, dirs, files in os.walk(input_folder):
        if os.path.basename(root) == "photo":
            image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
            if len(image_files) >= images_per_folder:
                selected_images = random.sample(image_files, images_per_folder)
                for image in selected_images:
                    src_path = os.path.join(root, image)
                    dest_path = os.path.join(output_folder, f"{count}_{image}")
                    shutil.copy(src_path, dest_path)
                    count += 1


# Example usage:
input_folder = "/home/linuxu/yuri/data/SceneNet_RGB-D/train_0"
output_folder = "/home/linuxu/yuri/data/SceneNet_RGB-D/smaller_combined_train_0_data"
copy_random_images(input_folder, output_folder)
