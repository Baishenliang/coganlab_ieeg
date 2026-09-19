# LexicalDelay bipolar physiology × ESM reproducibility bundle

This package maps the **exact bipolar LexicalDelay physiological classification**
onto ESM-tested bipolar pairs and reproduces all summary tables, models, and figures.

## Source data

- `data/bipolar_physiology.csv`
  - User-provided exact bipolar physiology classifications.
  - Fields used: `liang_encoding`, `liang_motor_preparation`, `liang_delay`,
    `liang_SM_vWM`, and `liang_class`.

- `data/lexdelay_esm_sitelevel.csv`
  - Clean LexicalDelay ESM site-level table used in the prior ESM analysis.
  - Includes `strict_esm_pos` and the pre-existing `broad_anatomy` classification.

No approximate reconstruction of the paper physiology classes is used here.

## Run

```bash
python scripts/01_run_analysis.py
python scripts/02_make_figures.py
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

## Matched dataset

The exact bipolar match yields:

- **96 ESM-tested bipolar pairs**
- **21 language-positive**
- **75 language-negative**
- **9 participants**

The original ESM table contains 193 clean LexicalDelay pairs. Ninety-six have an
exact bipolar physiology match in the supplied physiology file. Coverage statistics
are saved in `outputs/coverage_summary.csv`.

## Analyses

### 1. Exact physiology-class rates

`outputs/class_rates.csv`

ESM+ rates are calculated for the exact classes:
- Sensory-motor vWM
- Auditory vWM
- Motor vWM
- Delay-only
- Sensory-motor no-vWM
- Auditory no-vWM
- Motor no-vWM
- Other / unclassified

Wilson 95% confidence intervals are provided.

### 2. Primary class contrasts

`outputs/primary_contrasts.csv`

Tests:
- Sensory-motor vWM vs all other classes
- Any vWM (`liang_delay=True`) vs no-vWM/unclassified
- Sensory-motor vWM vs Auditory vWM
- Sensory-motor vWM vs all other vWM classes

For each contrast the output includes:
- raw ESM+ rates
- pooled Fisher exact OR and p
- macro within-subject AUC
- one-sided subject-preserving permutation p
- subject-fixed conditional-logit OR
- subject×anatomy conditional-logit OR

### 3. Feature-level analysis

`outputs/feature_associations.csv`

Binary features tested:
- encoding
- motor preparation
- delay
- sensory-motor vWM

The key distinction is between pooled associations and associations that survive
within-subject / anatomy adjustment.

### 4. Joint model

`outputs/joint_conditional_models.csv`

Conditional logistic models include:
- encoding
- motor preparation
- delay

simultaneously, first stratified by subject and then by subject × broad anatomy.

## Key interpretation

The **raw exact-class result is strong for Sensory-motor vWM**, but most of those
sites come from one participant (D0081), so the pooled class association attenuates
substantially after within-subject adjustment.

The more robust result is **motor preparation**:

- pooled ESM association is large;
- the within-subject permutation test is significant;
- the subject-fixed conditional model is significant;
- the subject×anatomy conditional model remains significant in the univariate analysis;
- in the joint subject-fixed model containing encoding + motor preparation + delay,
  motor preparation remains the dominant predictor.

This means the strongest supported causal relationship in this exact bipolar
classification is with **pre-articulatory/motor-preparation physiology**, rather
than delay activity alone.

## Figures

- `Figure1_ESM_rate_by_exact_physiology_class.png`
- `Figure2_feature_level_ESM_associations.png`
- `Figure3_conditional_OR_forest.png`
- `Figure4_motor_preparation_by_participant.png`


### Reproducibility note
The supplied script uses 10,000 fixed-seed within-subject permutations so the reported Monte Carlo p-values reproduce exactly while running quickly on a standard laptop.
