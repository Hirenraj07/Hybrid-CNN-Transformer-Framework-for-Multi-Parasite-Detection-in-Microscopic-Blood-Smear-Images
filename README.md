# Hybrid-CNN-Transformer-Framework-for-Multi-Parasite-Detection-in-Microscopic-Blood-Smear-Images
This project presents a deep learning framework for automated multi-parasite classification using microscopic blood smear images.

# 1. Introduction

Microscopic examination of stained blood smears is a primary diagnostic method for detecting hematological parasites. However, manual analysis is labor-intensive, requires expert interpretation, and may suffer from inter-observer variability.

This project develops an automated multi-class parasite detection system using a hybrid deep learning architecture. The framework combines EfficientNet-B0 for feature extraction with a transformer encoder to model global contextual relationships between spatial features.

The system is designed to be both accurate and explainable, supporting potential clinical decision assistance.

# 2. Dataset Description

The dataset consists of 34,298 microscopic images captured under 400X and 1000X magnification. It includes eight classes:

Babesia

Leishmania

Leukocyte

Plasmodium

RBCs

Toxoplasma

Trichomonad

Trypanosome

The dataset includes both infected and non-infected host cells to ensure realistic multi-class classification.

The dataset is not included in this repository due to size constraints. Users must obtain and organize it separately.

# 3. Problem Statement

Multi-parasite detection presents several challenges:

Morphological similarity between parasite species

Class imbalance

Subtle intra-class variations

Risk of overfitting in high-capacity models

Lack of interpretability in deep learning systems

This project addresses these challenges using transfer learning, hybrid architecture design, controlled fine-tuning, and explainable AI techniques.

# 4. Methodology
4.1 Data Preprocessing

Images resized to 224 × 224 pixels

EfficientNet-specific preprocessing applied

Stratified train–validation–test split

Controlled handling of class imbalance

Proper preprocessing alignment with pretrained EfficientNet weights was critical for stable training and high performance.

4.2 Baseline Model: EfficientNet-B0

Baseline architecture:

EfficientNet-B0 → Global Average Pooling → Dense Layer → Softmax (8 classes)

The baseline model achieved approximately 99% test accuracy but showed relatively lower F1-score for the minority class Plasmodium.

4.3 Proposed Hybrid CNN–Transformer Model

Hybrid architecture:

EfficientNet-B0 (feature extractor)
→ Feature reshaping into token sequence
→ Multi-Head Self-Attention (Transformer Encoder Block)
→ Feed-forward network
→ Global pooling
→ Dense layer
→ Softmax classifier

The transformer encoder enables modeling of global dependencies across spatial feature representations extracted by the CNN backbone.

4.4 Fine-Tuning Strategy

Fine-tuning was performed using:

Partial unfreezing of backbone layers

Low learning rate (5e-5)

Gradient clipping

ReduceLROnPlateau scheduling

Early stopping

Label smoothing

This controlled approach improved minority-class discrimination without overfitting.

# 5. Experimental Results
Model Performance Comparison
Model	Test Accuracy	Macro F1	Plasmodium F1
EfficientNet-B0	~99%	0.98	0.91
Hybrid (Fine-tuned)	~99.6%	0.99	0.95

Key observations:

The hybrid architecture improves minority-class performance.

Plasmodium F1-score increased from 0.91 to 0.95.

Macro-average metrics improved, indicating better class balance handling.

# 6. Explainability Using Grad-CAM

Grad-CAM was implemented to visualize spatial regions contributing to model predictions.

Observations:

Heatmaps highlight parasite morphology within infected cells.

Background regions are largely suppressed.

The hybrid model demonstrates sharper and more localized attention compared to the baseline.

These findings indicate that model predictions align with hematological diagnostic patterns.

# 7. Repository Structure

Parasite_Detection/

parasite_detection.ipynb

requirements.txt

README.md

models/

baseline_efficientnet_b0.keras

hybrid_efficientnet_vit_finetuned_99.keras

outputs/

confusion_matrix.png

gradcam_samples/

dataset/

(Dataset not included)

# 8. Installation

Install dependencies using:

pip install -r requirements.txt

Required packages:

tensorflow

numpy

pandas

scikit-learn

opencv-python

matplotlib

# 9. Reproducibility

To reproduce results:

Organize dataset into class-wise folders.

Run the notebook sequentially.

Train baseline model.

Train hybrid model.

Perform fine-tuning.

Generate Grad-CAM visualizations.

Ensure EfficientNet preprocessing is applied consistently during both training and inference.

# 10. Contributions

Development of hybrid CNN–Transformer architecture

Minority-class performance improvement

Controlled fine-tuning strategy

Integration of Grad-CAM for interpretability

Comprehensive evaluation using class-wise metrics

# 11. Limitations

Dataset limited to single-source images

No external validation dataset

Real-time deployment not implemented

Future work may include multi-center validation and deployment-ready system design.

# 12. Conclusion

The proposed hybrid CNN–Transformer framework improves minority-class discrimination in multi-parasite blood smear classification tasks while maintaining high overall accuracy. Grad-CAM analysis confirms biologically meaningful attention patterns, supporting the interpretability and clinical relevance of the system.


