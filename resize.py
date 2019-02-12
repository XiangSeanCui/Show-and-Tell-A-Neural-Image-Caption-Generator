import os
from PIL import Image


def resize_images(image_dir, output_dir, size):
    '''Resize all images in image_dir, and save to output_dir with given size'''
    # Check if output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)  # return a list of images name
    num_images = len(images)  # total number of images
    i = 0
    for img_name in images:
        with open(os.path.join(image_dir, img_name), 'rb') as f:
            with Image.open(f) as image:
                image = image.resize(size, Image.ANTIALIAS)
                image.save(os.path.join(output_dir, img_name), image.format)
        if (i+1) % 100 == 0:
            print('Resized the %d image out of  %d' % (i+1, num_images))
        i += 1
    print('Resize done!, Saved to', output_dir)


if __name__ == '__main__':
    image_dir = './data/train2014'
    output_dir = './data/resized2014_temp'
    image_dim = [256, 256]
    resize_images(image_dir, output_dir, image_dim)
