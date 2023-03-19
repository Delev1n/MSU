import os
from PIL import Image
import sys


def combine_12_leads(base: str):

    lead_names = ['I_lead', 'II_lead', 'III_lead', 'aVR_lead', 'aVL_lead', 'aVF_lead',
                  'V1_lead', 'V2_lead', 'V3_lead', 'V4_lead', 'V5_lead', 'V6_lead']

    full_path = base + '/combined/'
    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    file_names = os.listdir(base + '/I_lead/')

    for j, file_name in enumerate(file_names):
        images = []
        for lead in lead_names:
            images.append(Image.open("{}/{}/{}".format(base, lead, file_name)).convert('RGB'))
        new_image = Image.new('RGB', (2 * images[0].size[0], 6 * images[0].size[1]))
        for i, img in enumerate(images):
            if int(i / 5) == 0 or i == 5:
                size_x = 0
                size_y = i * img.size[1]
            else:
                size_x = img.size[0]
                size_y = (i - 6) * img.size[1]
            new_image.paste(img, (size_x, size_y))
        new_image.save("{}{}".format(full_path, file_name), "JPEG")
        print("{}/{}".format(j + 1, len(file_names)), end="\r")


if __name__ == "__main__":

    try:
        base = sys.argv[1]
    except:
        raise Exception("Please input the following data:\nBase directory, that contains 12 folders with pictures"
                        "of each lead")

    combine_12_leads(base)
