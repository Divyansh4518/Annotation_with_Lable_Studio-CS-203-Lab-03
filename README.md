# Annotation_with_Lable_Studio-CS-203-Lab-03

# Dataset Annotation and Inter-Annotator Agreement Calculation

This repository contains the code, annotations, and documentation for a lab assignment focused on dataset annotation and inter-annotator agreement calculation. The project involves annotating Hindi-English datasets using Label Studio, calculating inter-annotator agreement scores (Cohen’s Kappa and Fleiss Kappa), and interpreting the results.

---

## Project Overview

### Tasks Completed:
1. **Environment Setup**:
   - Installed Python 3.10 using a tarball.
   - Created a Pip environment and installed Label Studio.
   - Verified the setup with screenshots and command history.

2. **Dataset Annotation**:
   - Annotated 20 data points from both NLP and CV datasets using Label Studio.
   - Applied POS tags and NER for the NLP dataset.
   - Classified images as "Truck" or "No Truck" for the CV dataset.

3. **Inter-Annotator Agreement Calculation**:
   - Exported annotations in JSON/CSV format.
   - Calculated Cohen’s Kappa for the NLP dataset.
   - Calculated Fleiss Kappa for the CV dataset with a third annotator.

---

## Repository Structure

project/
├── annotations/                  # Exported annotation files (JSON/CSV)
│   ├── nlp_annotations.csv
│   ├── cv_annotations.csv
├── code/                         # Python scripts for calculations
│   ├── cohens_kappa.py
│   ├── fleiss_kappa.py
├── screenshots/                  # Screenshots of environment setup
│   ├── history_cleared.png
│   ├── label_studio_setup.png
├── user_history.txt              # Command history for Task 1
├── README.md                     # This file


---

## How to Run the Code

### Prerequisites
- Python 3.10
- Pip
- Label Studio
- Libraries: `scikit-learn`, `pandas`, `statsmodels`

### Steps:
1. Clone the repository:
   bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Run the scripts:
   - For Cohen’s Kappa (NLP dataset):
     bash
     python code/cohens_kappa.py
     
   - For Fleiss Kappa (CV dataset):
     bash
     python code/fleiss_kappa.py
     

---

## Results

### Inter-Annotator Agreement Scores:
- **Cohen’s Kappa (NLP Dataset)**: `0.85` (Excellent agreement)
- **Fleiss Kappa (CV Dataset)**: `0.78` (Good agreement)

### Interpretation:
- High scores indicate consistent annotations among team members.
- Minor disagreements were resolved by revisiting annotation guidelines.

---

## Screenshots

1. **Environment Setup**:
   - ![History Cleared](screenshots/history_cleared.png)
   - ![Label Studio Setup](screenshots/label_studio_setup.png)

2. **Annotation Examples**:
   - NLP Dataset: POS tagging and NER.
   - CV Dataset: Image classification.

---

## Submission Details

- **GitHub Repository**: [Link to your repository](#)
- **Google Form Submission**: [Link to submission form](#)

---

## Honor Code

By submitting this assignment, we confirm that we have followed IITGN's honor code. All annotations and code are original, and any external references have been properly cited.

---
## Team Number - 11
---
## Team Members
1. [Divyansh Saini]
2. [Gaurav Srivastava]


