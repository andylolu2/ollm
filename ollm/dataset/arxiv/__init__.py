ROOT_CATEGORY_ID = "Main topic classifications"
ROOT_CATEGORY_NAME = "Main topic classifications"

ALIASES = [
    ["math.MP", "math-ph"],
    ["stat.TH", "math.ST"],
    ["math.IT", "cs.IT"],
    ["econ.GN", "q-fin.EC"],
    ["cs.SY", "eess.SY"],
    ["cs.NA", "math.NA"],
    ["physics", "grp_physics"],
    ["econ", "grp_econ"],
    ["math", "grp_math"],
    ["q-bio", "grp_q-bio"],
    ["q-fin", "grp_q-fin"],
    ["cs", "grp_cs"],
    ["stat", "grp_stat"],
    ["eess", "grp_eess"],
]


def normalise(category_id):
    for aliases in ALIASES:
        if category_id in aliases:
            return aliases[0]
    return category_id
