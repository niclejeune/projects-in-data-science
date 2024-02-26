The PAD-UFES-20 dataset, collected as part of the Dermatological and Surgical Assistance Program at the Federal University of Espírito Santo, Brazil, includes 2,298 samples of six types of skin lesions. The lesions are categorized into three skin cancers (Basal Cell Carcinoma, Melanoma, Squamous Cell Carcinoma) and three skin diseases (Actinic Keratosis, Nevus, Seborrheic Keratosis). Bowen’s disease, considered a form of Squamous Cell Carcinoma, is included in this classification. About 58% of the samples are biopsy-proven (sample of the lesion tested in a laboratory). 

The dataset contains clinical images and their features in the metadata, including age, lesion location, skin type, and diameter. The dataset involves 1,373 patients, 1,641 skin lesions, and is available in .png format, with metadata provided in CSV format detailing up to 26 features per lesion​​. The images were taken with different smartphones, so the images come in different sizes and various quality. 

Basal Cell Carcinoma (BCC), Melanoma, and Squamous Cell Carcinoma (SCC) are three types of skin cancer related to UV exposure. BCC is the most common and least aggressive, while Melanoma is aggressive and arises from pigment-producing cells. SCC is less aggressive than Melanoma but can become invasive.

Actinic Keratosis is a precancerous condition, Nevus (mole) can become cancerous, and Seborrheic Keratosis is benign. Bowen's disease is a precancerous form of SCC. Regular skin checks and sun protection are essential for prevention and early detection.

A full list of features which are always available in the metadata is:
- Patient id
- Lesion id
- Age
- Region (where the lesion is located on the body)
- Diagnostic (which category it is, out of the 3 skin cancers and 3 diseases)
- Itch (whether the lesion itches)
- Grew (if the lesion grew or not, uncertain for some)
- Hurt (if it hurts)
- Changed (if it’s changed)
- Bleed (if the lesion has bled)
- Elevation (if there’s a “bump”)
- The image id (file)
- If the lesion has been biopsied

Some features are only available for some images (missing data):
- Smoke (if the patient if a smoker)
- Drink (if the patient is a drinker)
- Background father/mother (country of origin of parents)
- Pesticide
- Gender
- If the patient has a history of skin cancer
- If the patient has a history of cancer
- If the patient’s house has a water mains connection (piped water)
- If the patient’s house has a sewage system connection
- Skin color (on Fitzpatrick scale)
- Lesion diameter (2 axis, mm)
