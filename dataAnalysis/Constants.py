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

VARIATION_DF = "variation.csv"
VARIATION_SMALL_DF = "variation_small.csv"

STEPS = 20 # for HetGNN 14 to reduce computational complexity; otherwise 20

FEATURES = [AGE_COLUMN_NAME, SEX_CATEGORY_COLUMN_NAME, HGB_COLUMN_NAME, WBC_COLUMN_NAME, RBC_COLUMN_NAME, MCV_COLUMN_NAME, PLT_COLUMN_NAME]
#              ,'Noise0', 'Noise1', 'Noise2', 'Noise3', 'Noise4', 'Noise5', 'Noise6', 'Noise7', 'Noise8', 'Noise9'
# , 'Noise10', 'Noise11', 'Noise12', 'Noise13', 'Noise14', 'Noise15', 'Noise16', 'Noise17', 'Noise18', 'Noise19', 'Noise20', 'Noise21', 'Noise22', 'Noise23', 'Noise24', 'Noise25', 'Noise26', 'Noise27', 'Noise28', 'Noise29', 'Noise30', 'Noise31', 'Noise32', 'Noise33', 'Noise34', 'Noise35', 'Noise36', 'Noise37', 'Noise38', 'Noise39', 'Noise40', 'Noise41', 'Noise42', 'Noise43', 'Noise44', 'Noise45', 'Noise46', 'Noise47', 'Noise48', 'Noise49', 'Noise50', 'Noise51', 'Noise52', 'Noise53', 'Noise54', 'Noise55', 'Noise56', 'Noise57', 'Noise58', 'Noise59', 'Noise60', 'Noise61', 'Noise62', 'Noise63', 'Noise64', 'Noise65', 'Noise66', 'Noise67', 'Noise68', 'Noise69', 'Noise70', 'Noise71', 'Noise72', 'Noise73', 'Noise74', 'Noise75', 'Noise76', 'Noise77', 'Noise78', 'Noise79', 'Noise80', 'Noise81', 'Noise82', 'Noise83', 'Noise84', 'Noise85', 'Noise86', 'Noise87', 'Noise88', 'Noise89', 'Noise90', 'Noise91', 'Noise92', 'Noise93', 'Noise94', 'Noise95', 'Noise96', 'Noise97', 'Noise98', 'Noise99', 'Noise100', 'Noise101', 'Noise102', 'Noise103', 'Noise104', 'Noise105', 'Noise106', 'Noise107', 'Noise108', 'Noise109', 'Noise110', 'Noise111', 'Noise112', 'Noise113', 'Noise114', 'Noise115', 'Noise116', 'Noise117', 'Noise118', 'Noise119', 'Noise120', 'Noise121', 'Noise122', 'Noise123', 'Noise124', 'Noise125', 'Noise126', 'Noise127', 'Noise128', 'Noise129', 'Noise130', 'Noise131', 'Noise132', 'Noise133', 'Noise134', 'Noise135', 'Noise136', 'Noise137', 'Noise138', 'Noise139', 'Noise140', 'Noise141', 'Noise142', 'Noise143', 'Noise144', 'Noise145', 'Noise146', 'Noise147', 'Noise148', 'Noise149', 'Noise150', 'Noise151', 'Noise152', 'Noise153', 'Noise154', 'Noise155', 'Noise156', 'Noise157', 'Noise158', 'Noise159', 'Noise160', 'Noise161', 'Noise162', 'Noise163', 'Noise164', 'Noise165', 'Noise166', 'Noise167', 'Noise168', 'Noise169', 'Noise170', 'Noise171', 'Noise172', 'Noise173', 'Noise174', 'Noise175', 'Noise176', 'Noise177', 'Noise178', 'Noise179', 'Noise180', 'Noise181', 'Noise182', 'Noise183', 'Noise184', 'Noise185', 'Noise186', 'Noise187', 'Noise188', 'Noise189', 'Noise190', 'Noise191', 'Noise192', 'Noise193', 'Noise194', 'Noise195', 'Noise196', 'Noise197', 'Noise198', 'Noise199', 'Noise200', 'Noise201', 'Noise202', 'Noise203', 'Noise204', 'Noise205', 'Noise206', 'Noise207', 'Noise208', 'Noise209', 'Noise210', 'Noise211', 'Noise212', 'Noise213', 'Noise214', 'Noise215', 'Noise216', 'Noise217', 'Noise218', 'Noise219', 'Noise220', 'Noise221', 'Noise222', 'Noise223', 'Noise224', 'Noise225', 'Noise226', 'Noise227', 'Noise228', 'Noise229', 'Noise230', 'Noise231', 'Noise232', 'Noise233', 'Noise234', 'Noise235', 'Noise236', 'Noise237', 'Noise238', 'Noise239', 'Noise240', 'Noise241', 'Noise242', 'Noise243', 'Noise244', 'Noise245', 'Noise246', 'Noise247', 'Noise248', 'Noise249', 'Noise250', 'Noise251', 'Noise252', 'Noise253', 'Noise254', 'Noise255', 'Noise256', 'Noise257', 'Noise258', 'Noise259', 'Noise260', 'Noise261', 'Noise262', 'Noise263', 'Noise264', 'Noise265', 'Noise266', 'Noise267', 'Noise268', 'Noise269', 'Noise270', 'Noise271', 'Noise272', 'Noise273', 'Noise274', 'Noise275', 'Noise276', 'Noise277', 'Noise278', 'Noise279', 'Noise280', 'Noise281', 'Noise282', 'Noise283', 'Noise284', 'Noise285', 'Noise286', 'Noise287', 'Noise288', 'Noise289', 'Noise290', 'Noise291', 'Noise292', 'Noise293', 'Noise294', 'Noise295', 'Noise296', 'Noise297', 'Noise298', 'Noise299', 'Noise300', 'Noise301', 'Noise302', 'Noise303', 'Noise304', 'Noise305', 'Noise306', 'Noise307', 'Noise308', 'Noise309', 'Noise310', 'Noise311', 'Noise312', 'Noise313', 'Noise314', 'Noise315', 'Noise316', 'Noise317', 'Noise318', 'Noise319', 'Noise320', 'Noise321', 'Noise322', 'Noise323', 'Noise324', 'Noise325', 'Noise326', 'Noise327', 'Noise328', 'Noise329', 'Noise330', 'Noise331', 'Noise332', 'Noise333', 'Noise334', 'Noise335', 'Noise336', 'Noise337', 'Noise338', 'Noise339', 'Noise340', 'Noise341', 'Noise342', 'Noise343', 'Noise344', 'Noise345', 'Noise346', 'Noise347', 'Noise348', 'Noise349', 'Noise350', 'Noise351', 'Noise352', 'Noise353', 'Noise354', 'Noise355', 'Noise356', 'Noise357', 'Noise358', 'Noise359', 'Noise360', 'Noise361', 'Noise362', 'Noise363', 'Noise364', 'Noise365', 'Noise366', 'Noise367', 'Noise368', 'Noise369', 'Noise370', 'Noise371', 'Noise372', 'Noise373', 'Noise374', 'Noise375', 'Noise376', 'Noise377', 'Noise378', 'Noise379', 'Noise380', 'Noise381', 'Noise382', 'Noise383', 'Noise384', 'Noise385', 'Noise386', 'Noise387', 'Noise388', 'Noise389', 'Noise390', 'Noise391', 'Noise392', 'Noise393', 'Noise394', 'Noise395', 'Noise396', 'Noise397', 'Noise398', 'Noise399', 'Noise400', 'Noise401', 'Noise402', 'Noise403', 'Noise404', 'Noise405', 'Noise406', 'Noise407', 'Noise408', 'Noise409', 'Noise410', 'Noise411', 'Noise412', 'Noise413', 'Noise414', 'Noise415', 'Noise416', 'Noise417', 'Noise418', 'Noise419', 'Noise420', 'Noise421', 'Noise422', 'Noise423', 'Noise424', 'Noise425', 'Noise426', 'Noise427', 'Noise428', 'Noise429', 'Noise430', 'Noise431', 'Noise432', 'Noise433', 'Noise434', 'Noise435', 'Noise436', 'Noise437', 'Noise438', 'Noise439', 'Noise440', 'Noise441', 'Noise442', 'Noise443', 'Noise444', 'Noise445', 'Noise446', 'Noise447', 'Noise448', 'Noise449', 'Noise450', 'Noise451', 'Noise452', 'Noise453', 'Noise454', 'Noise455', 'Noise456', 'Noise457', 'Noise458', 'Noise459', 'Noise460', 'Noise461', 'Noise462', 'Noise463', 'Noise464', 'Noise465', 'Noise466', 'Noise467', 'Noise468', 'Noise469', 'Noise470', 'Noise471', 'Noise472', 'Noise473', 'Noise474', 'Noise475', 'Noise476', 'Noise477', 'Noise478', 'Noise479', 'Noise480', 'Noise481', 'Noise482', 'Noise483', 'Noise484', 'Noise485', 'Noise486', 'Noise487', 'Noise488', 'Noise489', 'Noise490', 'Noise491', 'Noise492', 'Noise493', 'Noise494', 'Noise495', 'Noise496', 'Noise497', 'Noise498', 'Noise499']
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