from cjo.base.stringconstants import unknown

category_abbreviation_dict = None
category_icons_dict = None
label_dict = None
default_perishable_categories = None
default_categories = None
default_categories_PNP = None
default_categories_P = None
default_categories_NP = None
NOT_A_PRODUCT = 'NOT A PRODUCT'


def format_as_currency(v):
    # TODO import it project-specific
    if v == '':
        return v
    return f'{v:.2f}'


try:
    from cjo_settings.__categories import __category_abbreviation_dict, __category_icons_dict, \
        default_perishable_categories, default_categories, default_categories_NP, default_categories_P, \
        default_categories_PNP
except ModuleNotFoundError:
    print('Please create a file ./settings/__categories.py that overrides the above')

try:
    from cjo_settings.__labels import label_dict
except ModuleNotFoundError:
    print('Please create a file ./settings/__labels.py that overrides the above')

# Add unknown, NO_PRODUCT abbreviations
category_abbreviation_dict = {**__category_abbreviation_dict,
                              NOT_A_PRODUCT: 'X',
                              unknown: 'UNK'}

# Add unknown, NO_PRODUCT images
category_icons_dict = {**__category_icons_dict,
                       NOT_A_PRODUCT: 'icons/freepik/Separate/notifications_error.png',
                       unknown: 'icons/freepik/Separate/signals-prohibitions_question.png'}

# No additional categories are perishable
default_perishable_categories = default_perishable_categories

uno_categories = [unknown, NOT_A_PRODUCT]

# These are the categories including unknown and NO_PRODUCT
default_categories_uno = default_categories + uno_categories

# These are derived values
category_choices = [(i, j) for i, j in category_abbreviation_dict.items()]
label_codes = list(label_dict.keys())
label_names = list(label_dict.values())
label_choices = [(i, j) for i, j in label_dict.items()]
labels_abbreviation_dict_unknown = {**label_dict, 'UNK': unknown}
label_choices_unknown = [(i, j) for i, j in labels_abbreviation_dict_unknown.items()]
