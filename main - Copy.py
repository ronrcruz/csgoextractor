import cv2
import tesserocr
import numpy as np
import os
from PIL import Image

def match_template(cropped_image, templates, threshold=0.8):
    for digit, template in templates.items():
        result = cv2.matchTemplate(cropped_image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        if len(loc[0]) > 0:
            return digit
    return None

def match_multiple_templates(cropped_image, templates, threshold=0.1):
    instances = {}

    # Define the color ranges for red and blue
    lower_red_range = np.array([0, 0, 120])
    upper_red_range = np.array([50, 50, 255])
    lower_blue_range = np.array([120, 0, 0])
    upper_blue_range = np.array([255, 50, 50])

    # Create masks for red and blue colors
    red_mask = cv2.inRange(cropped_image, lower_red_range, upper_red_range)
    blue_mask = cv2.inRange(cropped_image, lower_blue_range, upper_blue_range)

    for template_name, template in templates.items():
        # Determine if the template is an enemy or ally
        template_type = template_name.split('_')[0]

        # Apply the appropriate color mask to the cropped_image
        if template_type == 'enemy':
            cropped_image_color_filtered = cv2.bitwise_and(cropped_image, cropped_image, mask=blue_mask)
        elif template_type == 'ally':
            cropped_image_color_filtered = cv2.bitwise_and(cropped_image, cropped_image, mask=red_mask)

        # Perform template matching using cv2.TM_CCOEFF_NORMED method
        result = cv2.matchTemplate(cropped_image_color_filtered, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            instances[template_name] = 1
        else:
            instances[template_name] = 0

    return instances



def crop_and_ocr(image, crop_rectangles, api):
    cropped_ocr_results = []
    cropped_text_results = {}  # Dictionary to store text results
    for rect in crop_rectangles:
        x, y, w, h = rect["coordinates"]
        label = rect["label"]
        mask = rect.get("mask", None)  # Get the mask coordinates if specified
        cropped_image = image[y:y+h, x:x+w]      
        
        if label == "time":
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((2, 2), np.uint8)
            cropped_image = cv2.dilate(cropped_image, kernel, iterations=1)
            cropped_image = cv2.erode(cropped_image, kernel, iterations=1)
        
        elif label == "timer_first_digit":
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            matched_digit = match_template(cropped_image, templates)
            if matched_digit is not None:
                cropped_text_results[label] = matched_digit
                continue
        
        elif label == "entities":
            entities_count = match_multiple_templates(cropped_image, all_templates)

            ally_count = sum([count for template, count in entities_count.items() if template.startswith("ally_")])
            enemy_count = sum([count for template, count in entities_count.items() if template.startswith("enemy_")])

            cropped_text_results["allies"] = ally_count
            cropped_text_results["enemies"] = enemy_count

        elif label == "armor":
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        elif label == "map_location":
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)


        pil_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        api.SetImage(pil_cropped_image)
        text_result = api.GetUTF8Text().strip()
        cropped_text_results[label] = text_result  # Save the text result in the dictionary
        cropped_ocr_results.append((f"{label}.png", text_result))
        cv2.imwrite(f"{label}.png", cropped_image)

    return cropped_ocr_results, cropped_text_results

# Load the screenshot
image = cv2.imread("C:\\Users\\Ron\\Pictures\\vlcsnap-2023-04-16-20h42m09s629.png")
template_zero = cv2.imread("C:\\Users\\Ron\\Pictures\\digit_zero_template.png")
template_zero = cv2.cvtColor(template_zero, cv2.COLOR_BGR2GRAY)
template_one = cv2.imread("C:\\Users\\Ron\\Pictures\\digit_one_template.png")
template_one = cv2.cvtColor(template_one, cv2.COLOR_BGR2GRAY)
enemy_template_ct = cv2.imread("C:\\Users\\Ron\\Pictures\\enemy_template_ct.png")
ally_dead_normal = cv2.imread("C:\\Users\\Ron\\Pictures\\ally_dead_normal.png")
ally_dead_mic = cv2.imread("C:\\Users\\Ron\\Pictures\\ally_dead_mic.png")
ally_dead_green = cv2.imread("C:\\Users\\Ron\\Pictures\\ally_dead_green.png")

templates = {
    '0': template_zero,
    '1': template_one
}

all_templates = {
    'enemy_ct': enemy_template_ct,
    'ally_dead_normal': ally_dead_normal,
    'ally_dead_mic': ally_dead_mic,
    'ally_dead_green': ally_dead_green
}


# Define the crop rectangles as a list of dictionaries
crop_rectangles = [
    {"coordinates": (65, 1029, 95, 49), "label": "hp"},
    {"coordinates": (341, 1029, 95, 49), "label": "armor"},
    {"coordinates": (32, 29, 165, 23), "label": "map_location"},
    {"coordinates": (926, 4, 70, 22), "label": "time"},
    {"coordinates": (1648, 1030, 78, 40), "label": "clip"},
    {"coordinates": (1750, 1045, 49, 23), "label": "reserve"},
    {"coordinates": (926, 3, 21, 24), "label": "timer_first_digit"},
    {"coordinates": (544, 7, 819, 39), "label": "entities"},
    # Add more crop rectangles here if necessary
]

# Create a TesserOCR API instance with English language
with tesserocr.PyTessBaseAPI(lang='eng', path=r'C:\Program Files\Tesseract-OCR\tessdata') as api:
    # Perform OCR on the cropped images
    cropped_ocr_results, cropped_text_results = crop_and_ocr(image, crop_rectangles, api)

# Print the OCR results
for img_data in cropped_ocr_results:
    img_name, text_result = img_data  # Unpack the tuple correctly

# Access the text results using the labels
hp_text = cropped_text_results["hp"]
armor_text = cropped_text_results["armor"]
map_location_text = cropped_text_results["map_location"]
clip_text = cropped_text_results["clip"]
reserve_text = cropped_text_results["reserve"] 
ammo_text = cropped_text_results["clip"] + "/" + cropped_text_results["reserve"]
timer_first_digit = cropped_text_results["timer_first_digit"]
round_status = cropped_text_results["timer_first_digit"] + cropped_text_results["time"]

if not cropped_text_results["time"]:
    round_status = "Bomb is planted."
else:
    round_status = cropped_text_results["time"]

full_time = timer_first_digit + round_status
ally_count = cropped_text_results["allies"]
enemy_count = cropped_text_results["enemies"]


# Now you can use the text results in other parts of your code
print("HP:", hp_text)
print("Armor:", armor_text)
print("Map location:", map_location_text)
print("Timer first digit:", timer_first_digit)
print("Round status:", round_status)
print("Clip:", clip_text)
print("Reserve:", reserve_text)
print("Ammo:", ammo_text)
print("Full time:", full_time)
print("Ally count:", ally_count)
print("Enemy count:", enemy_count)
