SEX_COLUMN_NAME = "Sex"
SEX_CATEGORY_COLUMN_NAME = "SexCategory"
AGE_COLUMN_NAME = "Age"
HGB_COLUMN_NAME = "HGB"
WBC_COLUMN_NAME = "WBC"
RBC_COLUMN_NAME = "RBC"
MCV_COLUMN_NAME = "MCV"
PLT_COLUMN_NAME = "PLT"
CRP_COLUMN_NAME = "CRP"
LABEL_COLUMN_NAME = "Label"
PATIENT_NAME = "PATIENT"
EDGE_TYPE = "HAS"
REV_EDGE_TYPE = "rev_HAS"


VARIATION_DF = "../feature_variation/variation.csv"
VARIATION_SMALL_DF = "../feature_variation/variation_small.csv"

STEPS = 20 # for HetGNN 14 to reduce computational complexity; otherwise 20

FEATURES = [AGE_COLUMN_NAME, SEX_CATEGORY_COLUMN_NAME, HGB_COLUMN_NAME, WBC_COLUMN_NAME, RBC_COLUMN_NAME, MCV_COLUMN_NAME, PLT_COLUMN_NAME]
#           'F1',
#                    'F2',
#                    'F3',
#                    'F4',
#                    'F5',
#                    'F6',
#                    'F7',
#                    'F8',
#                    'F9',
#                    'F10','OF1']
# FEATURES = [HGB_COLUMN_NAME, WBC_COLUMN_NAME, RBC_COLUMN_NAME, MCV_COLUMN_NAME, PLT_COLUMN_NAME]

def replace_sex_column_name(feature):
    if feature == SEX_CATEGORY_COLUMN_NAME:
        return SEX_COLUMN_NAME
    return feature
FEATURES_IN_TABLE = list(map(replace_sex_column_name ,FEATURES)) ## initial feature name of sex inserted


FEATURE_DICT = {}
for value, key in enumerate(FEATURES):
    FEATURE_DICT[key] = value