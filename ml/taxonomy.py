"""Unified label space for the merged corpus.

Every source label maps to "Crop|kind|Condition" (kind: H healthy, D disease, P pest damage, N nutrient deficiency,
A abiotic / physiological) or to None when it is dropped. Rules used when merging across sources:
  * the same disease on the same crop gets ONE class even when sources name it differently (this also mixes lab and
    field photos inside a class, which weakens the "class == photo style" shortcut);
  * labels that are ambiguous against an existing class are dropped rather than guessed (reason in the comment);
  * non-leaf material (fruit, rhizome, dried/senescent leaves) is dropped.
The class key used everywhere else is  <Crop>___<Condition>  with spaces as underscores.
"""
KINDS = {"H": "healthy", "D": "disease", "P": "pest", "N": "deficiency", "A": "abiotic"}
NOT_PLANT, INSECT = "Not_a_plant___Not_a_plant", "Insect___Insect_pest"   # router classes of the leaf model


def parse(v):
    """'Tomato|D|Late blight' -> ('Tomato', 'disease', 'Late blight', 'Tomato___Late_blight')."""
    crop, kind, cond = v.split("|")
    return crop, KINDS[kind], cond, f"{crop}___{cond}".replace(" ", "_")


AGML_CITE = "mirrored by Project-AgML on Hugging Face"
SOURCES = {
    "plantvillage": dict(imagery="lab", license="CC BY-SA 3.0 (PlantVillage, Hughes & Salathe 2015)", labels={
        "Apple___Apple_scab": "Apple|D|Scab", "Apple___Black_rot": "Apple|D|Black rot",
        "Apple___Cedar_apple_rust": "Apple|D|Cedar apple rust", "Apple___healthy": "Apple|H|healthy",
        "Blueberry___healthy": "Blueberry|H|healthy",
        "Cherry_(including_sour)___healthy": "Cherry|H|healthy",
        "Cherry_(including_sour)___Powdery_mildew": "Cherry|D|Powdery mildew",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maize|D|Gray leaf spot",
        "Corn_(maize)___Common_rust_": "Maize|D|Common rust", "Corn_(maize)___healthy": "Maize|H|healthy",
        "Corn_(maize)___Northern_Leaf_Blight": "Maize|D|Leaf blight",
        "Grape___Black_rot": "Grape|D|Black rot", "Grape___Esca_(Black_Measles)": "Grape|D|Esca",
        "Grape___healthy": "Grape|H|healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape|D|Leaf blight",
        "Orange___Haunglongbing_(Citrus_greening)": "Citrus|D|Huanglongbing",
        "Peach___Bacterial_spot": "Peach|D|Bacterial spot", "Peach___healthy": "Peach|H|healthy",
        "Pepper,_bell___Bacterial_spot": "Bell Pepper|D|Bacterial spot", "Pepper,_bell___healthy": "Bell Pepper|H|healthy",
        "Potato___Early_blight": "Potato|D|Early blight", "Potato___healthy": "Potato|H|healthy",
        "Potato___Late_blight": "Potato|D|Late blight", "Raspberry___healthy": "Raspberry|H|healthy",
        "Soybean___healthy": "Soybean|H|healthy", "Squash___Powdery_mildew": "Squash|D|Powdery mildew",
        "Strawberry___healthy": "Strawberry|H|healthy", "Strawberry___Leaf_scorch": "Strawberry|D|Leaf scorch",
        "Tomato___Bacterial_spot": "Tomato|D|Bacterial spot", "Tomato___Early_blight": "Tomato|D|Early blight",
        "Tomato___healthy": "Tomato|H|healthy", "Tomato___Late_blight": "Tomato|D|Late blight",
        "Tomato___Leaf_Mold": "Tomato|D|Leaf mold", "Tomato___Septoria_leaf_spot": "Tomato|D|Septoria leaf spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato|P|Spider mites",
        "Tomato___Target_Spot": "Tomato|D|Target spot", "Tomato___Tomato_mosaic_virus": "Tomato|D|Mosaic virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato|D|Leaf curl virus"}),

    "plant_doc_classification": dict(imagery="field", license="CC BY 4.0 (PlantDoc)", labels={
        "Apple Scab Leaf": "Apple|D|Scab", "Apple leaf": "Apple|H|healthy", "Apple rust leaf": "Apple|D|Cedar apple rust",
        "Bell_pepper leaf": "Bell Pepper|H|healthy", "Bell_pepper leaf spot": "Bell Pepper|D|Bacterial spot",
        "Blueberry leaf": "Blueberry|H|healthy", "Cherry leaf": "Cherry|H|healthy",
        "Corn Gray leaf spot": "Maize|D|Gray leaf spot", "Corn leaf blight": "Maize|D|Leaf blight",
        "Corn rust leaf": "Maize|D|Common rust", "Peach leaf": "Peach|H|healthy",
        "Potato leaf early blight": "Potato|D|Early blight", "Potato leaf late blight": "Potato|D|Late blight",
        "Raspberry leaf": "Raspberry|H|healthy", "Soyabean leaf": "Soybean|H|healthy",
        "Squash Powdery mildew leaf": "Squash|D|Powdery mildew", "Strawberry leaf": "Strawberry|H|healthy",
        "Tomato Early blight leaf": "Tomato|D|Early blight", "Tomato Septoria leaf spot": "Tomato|D|Septoria leaf spot",
        "Tomato leaf": "Tomato|H|healthy", "Tomato leaf bacterial spot": "Tomato|D|Bacterial spot",
        "Tomato leaf late blight": "Tomato|D|Late blight", "Tomato leaf mosaic virus": "Tomato|D|Mosaic virus",
        "Tomato leaf yellow virus": "Tomato|D|Leaf curl virus", "Tomato mold leaf": "Tomato|D|Leaf mold",
        "Tomato two spotted spider mites leaf": "Tomato|P|Spider mites",
        "grape leaf": "Grape|H|healthy", "grape leaf black rot": "Grape|D|Black rot"}),

    # CCMT (Ghana): label is crop-agnostic, the 'crop' column disambiguates -> keys are "Crop/label"
    "crop_pest_disease_classification": dict(imagery="field", license="CC BY 4.0 (CCMT, Mensah et al. 2023)", crop_col="crop", labels={
        "Cashew/anthracnose": "Cashew|D|Anthracnose", "Cashew/gumosis": "Cashew|D|Gummosis",
        "Cashew/healthy": "Cashew|H|healthy", "Cashew/leaf miner": "Cashew|P|Leaf miner",
        "Cashew/red rust": "Cashew|D|Red rust",
        "Cassava/bacterial blight": "Cassava|D|Bacterial blight", "Cassava/brown spot": "Cassava|D|Brown spot",
        "Cassava/green mite": "Cassava|P|Green mite", "Cassava/healthy": "Cassava|H|healthy",
        "Cassava/mosaic": "Cassava|D|Mosaic disease",
        "Maize/fall armyworm": "Maize|P|Fall armyworm", "Maize/grasshoper": "Maize|P|Grasshopper",
        "Maize/healthy": "Maize|H|healthy", "Maize/leaf beetle": "Maize|P|Leaf beetle",
        "Maize/leaf blight": "Maize|D|Leaf blight",
        "Maize/leaf spot": "Maize|D|Gray leaf spot",        # field "leaf spot" pooled with gray (Cercospora) leaf spot
        "Maize/streak virus": "Maize|D|Streak virus",
        "Tomato/healthy": "Tomato|H|healthy",
        "Tomato/leaf blight": "Tomato|D|Early blight",      # lowland-tropics "leaf blight" = Alternaria (early) blight
        "Tomato/leaf curl": "Tomato|D|Leaf curl virus", "Tomato/septoria leaf spot": "Tomato|D|Septoria leaf spot",
        "Tomato/verticulium wilt": "Tomato|D|Verticillium wilt"}),

    # TOM2024 (West Africa, ~200 px field crops). Fruit-only, tiny (<100) and generic "abiotic" classes are dropped.
    "TOM2024_disease_classification": dict(imagery="field", license="CC BY 4.0 (TOM2024)", crop_col="crop_type", labels={
        "Maize/curvulariosis-d": "Maize|D|Curvularia leaf spot", "Maize/healthy-leaf": "Maize|H|healthy",
        "Maize/helminthosporiosis-d": "Maize|D|Leaf blight", "Maize/rust-d": "Maize|D|Common rust",
        "Maize/spodoptera-frugiperda-a": "Maize|P|Fall armyworm", "Maize/spodoptera-frugiperda-p": "Maize|P|Fall armyworm",
        "Maize/stripe-d": "Maize|D|Streak virus",
        "Onion/alternaria-d": "Onion|D|Purple blotch",                # Alternaria porri
        "Onion/caterpillar-p": "Onion|P|Caterpillar", "Onion/fusarium-d": "Onion|D|Fusarium basal rot",
        "Onion/healthy-leaf": "Onion|H|healthy", "Onion/virosis-d": "Onion|D|Viral disease",
        "Tomato/alternaria-d": "Tomato|D|Early blight", "Tomato/bacterial-floundering-d": "Tomato|D|Bacterial wilt",
        "Tomato/healthy-leaf": "Tomato|H|healthy", "Tomato/helicoverpa-armigera-p": "Tomato|P|Fruit borer",
        "Tomato/mite-d": "Tomato|P|Spider mites", "Tomato/tomato-late-blight-d": "Tomato|D|Late blight",
        "Tomato/virosis-d": "Tomato|D|Leaf curl virus"}),
    # Looked at and rejected: PFSD_Musa banana (pre-augmented flips/rotations -> leakage, mostly pseudostem photos);
    # maize_nitrogen_deficiency (whole-canopy plot photos, not leaves).

    "icassava": dict(imagery="field", license="unstated at source (iCassava 2019 / TFDS 'cassava')", labels={
        "cbb": "Cassava|D|Bacterial blight", "cbsd": "Cassava|D|Brown streak disease", "cgm": "Cassava|P|Green mite",
        "cmd": "Cassava|D|Mosaic disease", "healthy": "Cassava|H|healthy"}),

    "paddy_disease_classification": dict(imagery="field", license="unstated on mirror (Paddy Doctor, Petchiammal et al. 2023)", labels={
        "bacterial_leaf_blight": "Rice|D|Bacterial leaf blight", "bacterial_leaf_streak": "Rice|D|Bacterial leaf streak",
        "bacterial_panicle_blight": "Rice|D|Bacterial panicle blight", "blast": "Rice|D|Blast",
        "brown_spot": "Rice|D|Brown spot", "dead_heart": "Rice|P|Stem borer (dead heart)",
        "downy_mildew": "Rice|D|Downy mildew", "hispa": "Rice|P|Hispa", "normal": "Rice|H|healthy",
        "tungro": "Rice|D|Tungro"}),
    "rice_leaf_bacterial_fungal": dict(imagery="mixed", license="CC BY 4.0", labels={
        "Bacterial Leaf Blight": "Rice|D|Bacterial leaf blight", "Brown Spot": "Rice|D|Brown spot",
        "Healthy Rice Leaf": "Rice|H|healthy", "Leaf Blast": "Rice|D|Blast", "Leaf scald": "Rice|D|Leaf scald",
        "Narrow Brown Leaf Spot": "Rice|D|Narrow brown leaf spot", "Rice Hispa": "Rice|P|Hispa",
        "Sheath Blight": "Rice|D|Sheath blight"}),
    "rice_leaf_disease_classification_india": dict(imagery="field", license="CC BY 4.0 (Sethy et al. 2020)", labels={
        "Bacterialblight": "Rice|D|Bacterial leaf blight", "Blast": "Rice|D|Blast", "Brownspot": "Rice|D|Brown spot",
        "Tungro": "Rice|D|Tungro"}),

    "sugarcane_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0 (Thite et al. 2024)", labels={
        "Banded Chlorosis": "Sugarcane|A|Banded chlorosis", "Brown Spot": "Sugarcane|D|Brown spot",
        "BrownRust": "Sugarcane|D|Brown rust", "Dried": None,              # senescent leaves, not a diagnosis
        "Grassy shoot": "Sugarcane|D|Grassy shoot", "Healthy": "Sugarcane|H|healthy",
        "Pokkah Boeng": "Sugarcane|D|Pokkah boeng", "Sett Rot": "Sugarcane|D|Sett rot", "Smut": "Sugarcane|D|Smut",
        "Viral Disease": "Sugarcane|D|Mosaic virus", "Yellow Leaf": "Sugarcane|D|Yellow leaf disease"}),

    "potato_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0 (Shabrina et al. 2023)", labels={
        "Bacteria": "Potato|D|Bacterial wilt", "Fungi": None,   # coarse "fungi" bucket would collide with early blight
        "Healthy": "Potato|H|healthy", "Nematode": "Potato|D|Nematode damage", "Pest": "Potato|P|Insect damage",
        "Phytopthora": "Potato|D|Late blight", "Virus": "Potato|D|Viral disease"}),

    "cotton_leaf_disease_classification_bangladesh_2": dict(imagery="mixed", license="CC BY 4.0 (SAR-CLD-2024)", labels={
        "Bacterial Blight": "Cotton|D|Bacterial blight", "Curl Virus": "Cotton|D|Leaf curl virus",
        "Healthy Leaf": "Cotton|H|healthy", "Herbicide Growth Damage": "Cotton|A|Herbicide damage",
        "Leaf Hopper Jassids": "Cotton|P|Jassids", "Leaf Redding": "Cotton|N|Leaf reddening",
        "Leaf Variegation": "Cotton|A|Leaf variegation"}),

    "groundnut_leaf_disease_classification_2": dict(imagery="field", license="CC BY 4.0", labels={
        "early_leaf_spot": "Groundnut|D|Early leaf spot", "healthy leaf": "Groundnut|H|healthy",
        "late leaf spot": "Groundnut|D|Late leaf spot", "nutrition deficiency": "Groundnut|N|Nutrient deficiency",
        "rust": "Groundnut|D|Rust"}),

    # the mirror stores bare integer labels; order checked on a contact sheet (0 spots, 1 healthy, 2 curling, ...)
    "COLD_chili_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0 (COLD)", labels={
        "0": "Chilli|D|Cercospora leaf spot", "1": "Chilli|H|healthy", "2": "Chilli|P|Mites and thrips",
        "3": "Chilli|N|Nutrient deficiency", "4": "Chilli|D|Powdery mildew"}),
    "COLD_onion_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0 (COLD)", labels={
        "Iris yellow virus": "Onion|D|Viral disease",                 # pooled with TOM2024 "virosis"
        "Stemphylium leaf blight and collectrichum leaf blight": "Onion|D|Leaf blight",
        "healthy": "Onion|H|healthy", "purple blotch": "Onion|D|Purple blotch"}),

    "mango_leaf_disease_classification": dict(imagery="lab", license="CC BY-NC 3.0 upstream (MangoLeafBD) - non-commercial", labels={
        "Anthracnose": "Mango|D|Anthracnose", "Bacterial_Canker": "Mango|D|Bacterial canker",
        "Cutting_Weevil": "Mango|P|Leaf-cutting weevil", "Die_Back": "Mango|D|Die back", "Gall_Midge": "Mango|P|Gall midge",
        "Healthy": "Mango|H|healthy", "Powdery_Mildew": "Mango|D|Powdery mildew", "Sooty_Mold": "Mango|D|Sooty mold"}),

    "banana_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "healthy": "Banana|H|healthy", "segatoka": "Banana|D|Sigatoka", "xamthomonas": "Banana|D|Xanthomonas wilt"}),
    "BananaLSD_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0 (BananaLSD)", labels={
        "cordana": "Banana|D|Cordana leaf spot", "healthy": "Banana|H|healthy",
        "pestalotiopsis": "Banana|D|Pestalotiopsis leaf spot", "sigatoka": "Banana|D|Sigatoka"}),
    "banana_leaf_nutrient_classification": dict(imagery="field", license="CC BY 4.0 (Sunitha et al.)", labels={
        "Boron": "Banana|N|Boron deficiency", "Calcium": "Banana|N|Calcium deficiency", "Healthy": "Banana|H|healthy",
        "Iron": "Banana|N|Iron deficiency", "Magnesium": "Banana|N|Magnesium deficiency",
        "Manganese": "Banana|N|Manganese deficiency", "Potassium": "Banana|N|Potassium deficiency",
        "Sulphur": "Banana|N|Sulphur deficiency", "Zinc": "Banana|N|Zinc deficiency"}),

    "citrusuat_disease_classification": dict(imagery="mixed", license="CC BY 4.0 (CitrusUAT)", labels={
        "Citrus_leafminer": "Citrus|P|Leafminer", "Fe": "Citrus|N|Iron deficiency", "Greasy_spot": "Citrus|D|Greasy spot",
        "HLB": "Citrus|D|Huanglongbing", "Healthy": "Citrus|H|healthy", "Mg": "Citrus|N|Magnesium deficiency",
        "Mn": "Citrus|N|Manganese deficiency", "N": "Citrus|N|Nitrogen deficiency", "Red_scale": "Citrus|P|Red scale",
        "Red_scale_sequelae": "Citrus|P|Red scale", "Texas_mite": "Citrus|P|Citrus mite",
        "Zn": "Citrus|N|Zinc deficiency"}),
    # Lemon stays a separate crop: its generic "Deficiency Leaf" would contradict the specific citrus deficiencies.
    "lemon_leaf_disease_classification": dict(imagery="mixed", license="CC BY 4.0", labels={
        "Anthracnose": "Lemon|D|Anthracnose", "Bacterial Blight": "Lemon|D|Bacterial blight",
        "Citrus Canker": "Lemon|D|Citrus canker", "Curl Virus": "Lemon|D|Leaf curl",
        "Deficiency Leaf": "Lemon|N|Nutrient deficiency", "Dry Leaf": None, "Healthy Leaf": "Lemon|H|healthy",
        "Sooty Mould": "Lemon|D|Sooty mould", "Spider Mites": "Lemon|P|Spider mites"}),

    "CoLeaf_nutritional_deficiency_classification": dict(imagery="lab", license="CC BY 4.0 (CoLeaf-DB)", labels={
        "boron-B": "Coffee|N|Boron deficiency", "calcium-Ca": "Coffee|N|Calcium deficiency", "healthy": "Coffee|H|healthy",
        "iron-Fe": "Coffee|N|Iron deficiency", "magnesium-Mg": "Coffee|N|Magnesium deficiency",
        "manganese-Mn": "Coffee|N|Manganese deficiency", "nitrogen-N": "Coffee|N|Nitrogen deficiency",
        "phosphorus-P": "Coffee|N|Phosphorus deficiency", "potasium-K": "Coffee|N|Potassium deficiency"}),

    "eggplant_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "Healthy Leaf": "Brinjal|H|healthy", "Insect Pest Disease": "Brinjal|P|Insect damage",
        "Leaf Spot Disease": "Brinjal|D|Leaf spot", "Mosaic Virus Disease": "Brinjal|D|Mosaic virus",
        "White Mold Disease": "Brinjal|D|White mold", "Wilt Disease": "Brinjal|D|Wilt"}),

    "BPLD_leaf_disease_classification": dict(imagery="mixed", license="CC BY 4.0 (BPLD, Krishna district, Andhra Pradesh)", labels={
        "Anthracnose": "Black gram|D|Anthracnose", "Healthy": "Black gram|H|healthy",
        "Leaf Crinckle": "Black gram|D|Leaf crinkle virus", "Powdery Mildew": "Black gram|D|Powdery mildew",
        "Yellow Mosaic": "Black gram|D|Yellow mosaic virus"}),
    "black_gram_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "Cercospora leaf spot": "Black gram|D|Cercospora leaf spot", "Healthy": "Black gram|H|healthy",
        "Insect": "Black gram|P|Insect damage", "Leaf Crinkle": "Black gram|D|Leaf crinkle virus",
        "Yellow Mosaic": "Black gram|D|Yellow mosaic virus"}),

    "teaLeafBD_disease_classification": dict(imagery="field", license="CC BY 4.0 (teaLeafBD)", labels={
        "Brown Blight": "Tea|D|Brown blight", "Gray Blight": "Tea|D|Gray blight", "Green mirid bug": "Tea|P|Green mirid bug",
        "Healthy leaf": "Tea|H|healthy", "Helopeltis": "Tea|P|Tea mosquito bug", "Red spider": "Tea|P|Red spider mite",
        "Tea algal leaf spot": "Tea|D|Algal leaf spot"}),

    "grape_leaf_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "Bacterial Leaf Spot": "Grape|D|Bacterial leaf spot", "Downy Mildew": "Grape|D|Downy mildew",
        "Healthy Leaves": "Grape|H|healthy", "Powdery Mildew": "Grape|D|Powdery mildew"}),
    "lentil_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "Ascochyta blight": "Lentil|D|Ascochyta blight", "Lentil Rust": "Lentil|D|Rust", "Normal": "Lentil|H|healthy",
        "Powdery Mildew": "Lentil|D|Powdery mildew"}),
    "bean_disease_uganda": dict(imagery="field", license="CC BY 4.0 (iBean, AIR Lab Makerere)", labels={
        "angular_leaf_spot": "Bean|D|Angular leaf spot", "bean_rust": "Bean|D|Rust", "healthy": "Bean|H|healthy"}),
    "IDDMSLD_spinach_leaf_disease_classification": dict(imagery="mixed", license="CC BY 4.0", labels={
        "Anthracnose": "Spinach|D|Anthracnose", "Bacterial-Spot": "Spinach|D|Bacterial spot",
        "Downy-Mildew": "Spinach|D|Downy mildew", "Healthy-Leaf": "Spinach|H|healthy",
        "Pest-Damage": "Spinach|P|Insect damage"}),
    "turmeric_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "Dry Leaf": None, "Healthy Leaf": "Turmeric|H|healthy", "Leaf Blotch": "Turmeric|D|Leaf blotch",
        "Rhizome Disease Root": None, "Rhizome Healthy Root": None}),       # rhizomes are not leaves
    "date_palm_leaf_disease_classification": dict(imagery="mixed", license="CC BY 4.0", labels={
        "Black Scorch": "Date palm|D|Black scorch", "Fusarium Wilt": "Date palm|D|Fusarium wilt",
        "Healthy sample": "Date palm|H|healthy", "Leaf Spots": "Date palm|D|Leaf spots",
        "Magnesium Deficiency": "Date palm|N|Magnesium deficiency",
        "Manganese Deficiency": "Date palm|N|Manganese deficiency",
        "Parlatoria Blanchardi": "Date palm|P|White scale", "Potassium Deficiency": "Date palm|N|Potassium deficiency",
        "Rachis Blight": "Date palm|D|Rachis blight"}),
    "betel_leaf_disease_classification_2": dict(imagery="mixed", license="CC BY 4.0", labels={
        "Healthy_Leaf": "Betel|H|healthy", "Leaf_Rot": "Betel|D|Leaf rot", "Leaf_Spot": "Betel|D|Leaf spot"}),

    # Zenodo 7573133 (Long et al. 2023), field + glasshouse; keys are the zip's folder names, matched case-insensitively
    "wheat_long2023": dict(imagery="mixed", license="CC BY 4.0 (Long et al. 2023, Zenodo 7573133)", labels={
        "yellow rust": "Wheat|D|Yellow rust", "brown rust": "Wheat|D|Brown rust", "septoria": "Wheat|D|Septoria leaf blotch",
        "mildew": "Wheat|D|Powdery mildew", "healthy": "Wheat|H|healthy"}),

    "papaya_leaf_disease_classification_bangladesh": dict(imagery="field", license="CC BY 4.0", labels={
        "Healthy Leaf": "Papaya|H|healthy", "Leaf Curl": "Papaya|D|Leaf curl virus", "Mealybug": "Papaya|P|Mealybug",
        "Mite Disease": "Papaya|P|Mites", "Mosaic": "Papaya|D|Mosaic virus", "Ring Spot": "Papaya|D|Ring spot virus"}),
    "MH_SoyaHealthVision_disease_classification_leaf": dict(imagery="field", license="CC BY 4.0 (MH-SoyaHealthVision, Maharashtra)", labels={
        "Caterpillar_Semilooper_Pest": "Soybean|P|Semilooper caterpillar", "Frog_Leaf_Eye": "Soybean|D|Frogeye leaf spot",
        "Healthy": "Soybean|H|healthy", "Mosaic": "Soybean|D|Mosaic virus", "Rust": "Soybean|D|Rust",
        "Spectoria_Brown_Spot": "Soybean|D|Septoria brown spot"}),
    "OkraDiseaseNet_disease_classification": dict(imagery="field", license="CC BY 4.0 (OkraDiseaseNet)", labels={
        "Alternaria Leaf Spot": "Okra|D|Alternaria leaf spot", "Cercospora Leaf Spot": "Okra|D|Cercospora leaf spot",
        "Downy Mildew": "Okra|D|Downy mildew", "Healthy": "Okra|H|healthy", "Leaf curly virus": "Okra|D|Leaf curl virus",
        "Phyllosticta leaf spot": "Okra|D|Phyllosticta leaf spot"}),
    "sunflower_disease_classification": dict(imagery="field", license="CC BY 4.0", labels={
        "Downy_mildew": "Sunflower|D|Downy mildew", "Fresh_leaf": "Sunflower|H|healthy", "Gray_mold": "Sunflower|D|Gray mold",
        "Leaf_scars": "Sunflower|D|Leaf scars"}),
}
