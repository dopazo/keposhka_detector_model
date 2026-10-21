"""
Grouped 2-fold cross-validation splits (data_split_cv0..1), so every labeled image is evaluated once by a model
that never saw it NOR other photos of the same sign (re-photos of the same rule screen go to the same fold).
Charts (alphabet / letra_*) are never evaluated: always in train, x3.

Usage:
    uv run python scripts/make_cv_folds.py
Then train each fold (see CLAUDE.md) and run scripts/cv_report.py.
"""

import subprocess
import sys

CHARTS = ("keposhka_abecedario keposhka_abecedario2 keposhka_abecedario_a_mano letra_CH letra_G letra_GH letra_J "
          "letra_L letra_Q letra_SH letra_TH letra_X letra_Z").split()
BREAKDOWN = ["breakdown-of-the-gentlemans-castle-video-during-teenagers-v0-p7hlkk9o4ghg1",
             "breakdown-of-the-gentlemans-castle-video-during-teenagers-v0-yqok11gt3ghg1"]

# "stage" = photos/screens of the MOAT PPAR rule signs (grouped by sign); "other" = everything else
STAGE = [
    ["rule_62", "20260214_205120", "20260214_203406", "20260214_202658"],
    ["rule_86", "rule_86_2", "20260214_202835", "rule_45", "rule_17", "20260214_202720"],
    ["rule_80", "20260214_202653", "rule_79", "rule_79_2", "20260214_205143", "rule_42"],
    ["rule_53", "rule_21", "rule_21_2"],
]
OTHER = [
    ["48emimjuxsnf1", "exercise_book_first_page", "Yea_nay"],
    BREAKDOWN + ["mcr_keposhka_guide", "jjmeny5u3gof1"],
    ["bi899cxrh9mf1", "postal_mcr_28ene", "postal_mcr_29ene", "postal_mcr_29ene_concierto", "logo_argentina"],
    ["setlist", "polera", "wanted"],
]

if __name__ == "__main__":
    for k in range(2):  # 2 folds: each tests half of the groups (a model trained on ~half the data: pessimistic)
        test = STAGE[2 * k] + STAGE[2 * k + 1] + OTHER[2 * k] + OTHER[2 * k + 1]
        subprocess.run([sys.executable, "scripts/prepare_holdout_split.py", "--out", f"data_split_cv{k}", "--test", *test,
                        "--oversample", "3", *CHARTS], check=True)
