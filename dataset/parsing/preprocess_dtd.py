
import os
import shutil

if __name__ == "__main__":
    base_dir = 'data'
    downloaded_data_path = f"{base_dir}/dtd/images"
    output_folder = f"{base_dir}/dtd/images"
    train_file = os.path.join(base_dir, 'dtd', 'labels', 'train1.txt')
    with open(train_file, 'r') as file:
        train_lines = file.readlines()
    with open(os.path.join(base_dir, 'dtd', 'labels', 'val1.txt'), 'r') as file:
        val_lines = file.readlines()
    with open(os.path.join(base_dir, 'dtd', 'labels', 'test1.txt'), 'r') as file:
        test_lines = file.readlines()

    for split in ['train', 'val', 'test']:
        if split == 'train':
            lines = train_lines
        elif split == 'val':
            lines = val_lines
        else:
            lines = test_lines
        output_folder = os.path.join(base_dir, 'dtd', split)
        os.makedirs(output_folder, exist_ok=True)
        for i, line in enumerate(lines):
            input_path = line.strip()

            # Extract folder name and filename
            final_folder_name = input_path.split('/')[:-1][0]
            filename = input_path.split('/')[-1]

            # Create output folder if it doesn't exist
            output_class_folder = os.path.join(output_folder, final_folder_name)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            # Copy file to the output folder
            full_input_path = os.path.join(downloaded_data_path, input_path)
            output_file_path = os.path.join(output_class_folder, filename)
            shutil.copy(full_input_path, output_file_path)

            # Print progress every 100 images
            if i % 100 == 0:
                print(f"Processed {i}/{len(lines)} images")
