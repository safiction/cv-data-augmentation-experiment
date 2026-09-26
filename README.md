# Synthetic Data Augmentation with Diffusion Models
## Project Question
Can diffusion-based synthetic data augmentation improve image classification performance when training data is limited, and under what conditions do synthetic samples help versus hurt?
## ML Task
Supervised image classification on a deliberately limited labeled dataset.
We compare:
• Baseline: classifier trained only on real images
• Augmented: the same classifier trained on real + diffusion-generated synthetic images for selected classes
Validation and test sets remain real-only, so performance is measured on real unseen data.
## Main Hypothesis
Adding high-quality diffusion-generated samples for underrepresented or data-scarce classes will improve classifier generalization on real unseen data compared with training on real data alone.
## Main Comparison
Real-only training vs. Real + synthetic training
Keep the following fixed:
• classifier architecture
• training procedure
• number of real training samples
• validation/test data
• evaluation metrics
## Metrics
• Macro F1
• Per-class precision
• Per-class recall
• Per-class F1
• Accuracy
Macro F1 is especially useful because each class contributes equally to the final score.
## Additional Research Question 1
How does the ratio of synthetic to real training samples affect classifier performance?
Possible ratios:
• 0% synthetic
• 25%
• 50%
• 100%
• 200%
This checks whether there is an optimal amount of synthetic augmentation and whether too much synthetic data starts to reduce performance.
## Additional Research Question 2
How do the quality and diversity of diffusion-generated images affect classifier performance?
Investigate whether artifacts, low diversity, or unrealistic synthetic features cause the classifier to learn synthetic-specific shortcuts instead of useful class features.


## Data description summary
train: 5094
validation: 900
test: 5794
Number of classes: 200

## Download the data
Run script:
```bash
python src/load_data.py
```

EDA is available in __notebooks/eda.ipynb__

Main observations:
- There is almost no class disbalance in the dataset, no missing images.
- Some pictures are in a grayscale, so need to convert them to RGB during training
- Image size is different, so need to resize or crop
- 2 duplicate images found