"""
NOT USED IN FINAL PIPELINE

Purpose:
Initial dataset inspection / sanity checking during
early experimentation with Coswara.

Kept for reproducibility and research transparency.
"""

import os
import csv

COSWARA_ROOT = "/Users/nafla/Desktop/FYP/Coswara-Data-master"
output_csv = "analysis_subject_inventory.csv"

rows = []

for outer_date in sorted(os.listdir(COSWARA_ROOT)):
    outer_path = os.path.join(COSWARA_ROOT, outer_date)
    if not os.path.isdir(outer_path):
        continue

    # Handle nested date folder (e.g. 20200413/20200413/)
    inner_date_path = os.path.join(outer_path, outer_date)
    if not os.path.isdir(inner_date_path):
        continue

    # Loop over subject folders
    for subject_id in os.listdir(inner_date_path):
        subject_path = os.path.join(inner_date_path, subject_id)
        if not os.path.isdir(subject_path):
            continue

        files = os.listdir(subject_path)

        has_deep = "breathing-deep.wav" in files
        has_shallow = "breathing-shallow.wav" in files

        total_breath_files = int(has_deep) + int(has_shallow)

        if total_breath_files > 0:
            rows.append([
                subject_id,
                outer_date,
                has_deep,
                has_shallow,
                total_breath_files
            ])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "subject_id",
        "date_folder",
        "has_deep",
        "has_shallow",
        "total_breath_files"
    ])
    writer.writerows(rows)

print("Done.")
print("Total subjects with breathing data:", len(rows))
print("Saved to:", output_csv)
