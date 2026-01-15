TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
CV_FOLDS_BINARY = 5
CV_FOLDS_MULTILABEL = 3

LANGUAGES = ["amh", "arb", "ben", "deu", "eng", "fas", "hau", "hin", "ita", "khm", "mya", 
             "nep", "ori", "pan", "pol", "rus", "spa", "swa", "tel", "tur", "urd", "zho"]

LANGUAGES_SUBTASK3 = ["amh", "arb", "ben", "deu", "eng", "fas", "hau", "hin", "khm",
                      "nep", "ori", "pan", "spa", "swa", "tel", "tur", "urd", "zho"]

SUBTASK_LANGUAGES = {
        1: LANGUAGES,
        2: LANGUAGES,
        3: LANGUAGES_SUBTASK3,
    }

SUBTASK1_LABELS = ["polarization"]

SUBTASK2_LABELS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

SUBTASK3_LABELS = ["stereotype", "vilification", "dehumanization", "extreme_language", "lack_of_empathy", "invalidation"]

SUBTASK_LABELS = {
        1: SUBTASK1_LABELS,
        2: SUBTASK2_LABELS,
        3: SUBTASK3_LABELS,
    }