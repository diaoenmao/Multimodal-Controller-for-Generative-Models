import os
import shutil
from utils import makedir_exist_ok


def main():
    img_folder = './output/img/test'
    filenames = os.listdir(img_folder)
    for filename in filenames:
        if os.path.isfile(os.path.join(img_folder, filename)):
            filename_head, filename_extension = os.path.splitext(filename)
            list_filename_head = filename_head.split('_')
            list_filename_head.pop(1)
            task = list_filename_head[0]
            subfolder = list_filename_head[-1]
            new_filename = '_'.join(list_filename_head) + filename_extension
            makedir_exist_ok(os.path.join(img_folder, task, subfolder))
            shutil.move(os.path.join(img_folder, filename), os.path.join(img_folder, task, subfolder, new_filename))


if __name__ == '__main__':
    main()