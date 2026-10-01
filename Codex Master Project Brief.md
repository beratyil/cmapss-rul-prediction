We are going to build a comprehensive machine-learning portfolio project together.

# Project

**NASA C-MAPSS Turbofan Remaining Useful Life Prediction**

The objective is to predict the Remaining Useful Life (RUL) of turbofan engines using multivariate time-series sensor measurements.

This project is NOT intended to be generated in one shot.

The primary objective is educational: I want to understand every important design decision and every important piece of the implementation so that I can explain and defend the project in a technical interview.

You are acting as my implementation partner.

## Core working rule

DO NOT implement the entire project at once.

We will work milestone by milestone.

Only work on the milestone that I explicitly request.

Do not silently implement future milestones.

Before making important architectural or machine-learning decisions, explain:

1. what problem we are solving,
2. the proposed solution,
3. why we are choosing it,
4. reasonable alternatives,
5. the trade-offs.

Keep explanations concise but technically meaningful.

## Language

- Source code: English
- Variable/function/class names: English
- README and repository documentation: English
- Git commit messages: English
- Explanations to me during development: Turkish

## Hardware

Development machine:

- NVIDIA RTX 3070 Ti
- 8 GB VRAM

Design training and inference accordingly.

Do not choose unnecessarily large models.

## Technology

Primary stack:

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- Plotly where interactive visualization is useful
- Streamlit later in the project

Add additional dependencies only when there is a clear reason.

## Dataset

Use the NASA C-MAPSS Jet Engine Simulated Data dataset.

Start with FD001.

Later milestones will extend the same pipeline to:

- FD002
- FD003
- FD004

Do not commit the raw dataset to Git.

Provide reproducible dataset acquisition/setup instructions and keep raw/processed datasets under gitignored directories.

## Repository philosophy

The final repository should look like a professional ML engineering project rather than a collection of notebooks.

Notebooks may be used for exploration and explanation, but important reusable functionality must eventually live under `src/`.

Suggested structure:

turbofan-rul/
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── turbofan_rul/
├── tests/
├── configs/
├── reports/
│   └── figures/
└── models/

Do not create unnecessary abstractions at the beginning.

Let the architecture evolve as the project becomes more complex.

## Planned milestones

### Milestone 0 — Project framing and repository setup

Define the problem clearly and create the minimal repository structure.

No machine-learning implementation yet.

### Milestone 1 — Dataset exploration

Load FD001 and perform exploratory data analysis.

Investigate:

- engine trajectories
- cycle counts
- operational settings
- sensor distributions
- sensor evolution over time
- constant and near-constant sensors
- correlations
- missing data
- potentially interesting degradation patterns

Produce clean and publication-quality visualizations.

### Milestone 2 — RUL target engineering

Implement and explain Remaining Useful Life calculation.

Compare:

- linear RUL
- piecewise/capped RUL

Explain why capped RUL targets are commonly considered.

Add automated tests for target generation.

### Milestone 3 — Classical baselines

Build simple baselines before deep learning.

Include suitable models such as:

- naive baseline
- linear or Ridge regression
- tree/boosting based baseline if appropriate

Establish meaningful baseline metrics.

### Milestone 4 — Time-series windowing

Create sequence samples suitable for neural networks.

Explain and test:

- sequence length
- features
- targets
- train/validation separation
- scaling
- data leakage
- tensor shapes

Add automated tests for sequence generation.

### Milestone 5 — LSTM model

Implement a deliberately simple LSTM regression model first.

Explain:

- input dimensions
- hidden state
- cell state
- hidden size
- number of layers
- output selection
- regression head
- loss function
- optimizer
- training loop

Do not optimize aggressively yet.

### Milestone 6 — GRU model

Implement GRU using the same training/evaluation pipeline.

Compare LSTM and GRU fairly.

Include:

- accuracy metrics
- parameter count
- training time
- memory requirements when practical

### Milestone 7 — Transformer model

Implement a reasonably small Transformer Encoder model for time-series RUL prediction.

Explain:

- input projection
- positional information
- Query, Key and Value
- self-attention
- multi-head attention
- Transformer Encoder
- regression head

Keep the model suitable for 8 GB VRAM.

### Milestone 8 — Experimental comparison and error analysis

Compare all models.

At minimum consider:

- RMSE
- MAE
- prediction plots
- engine-level error distributions
- early-life vs late-life prediction behavior
- training behavior
- model complexity

Focus on explaining WHY results differ, not only reporting scores.

### Milestone 9 — Generalization

Extend the pipeline from FD001 to FD002, FD003 and FD004.

Investigate performance degradation caused by:

- different operational conditions
- multiple fault modes
- distribution shift

The objective is not merely improving the score; it is understanding generalization.

### Milestone 10 — Explainability and ablation

Study what the models are using.

Potential techniques include:

- feature importance
- permutation importance
- SHAP where appropriate
- sensor ablation
- sequence-length ablation
- model component ablation

Do not claim that attention weights automatically constitute model explanations.

### Milestone 11 — Visualization application

Build a polished Streamlit dashboard.

Possible views:

- engine selection
- sensor history
- health/degradation visualization
- predicted Remaining Useful Life
- actual vs predicted RUL
- model comparison
- error analysis

Visual quality matters.

### Milestone 12 — ML engineering

Convert the project into a reproducible ML application.

Gradually add appropriate engineering features such as:

- configuration management
- experiment reproducibility
- deterministic seeds where possible
- structured logging
- tests
- CLI
- inference pipeline
- model serialization
- Docker
- GitHub Actions
- ONNX export if useful

Avoid adding infrastructure purely for appearance.

## Git discipline

Each milestone should correspond to one or more small, understandable commits.

At the end of every implementation step provide:

1. changed files,
2. what each changed file does,
3. how to run it,
4. expected output,
5. concepts I should understand before continuing,
6. unresolved technical questions,
7. a suggested Git commit message.

Do not commit generated models, raw datasets, caches, virtual environments or other large artifacts unless explicitly justified.

## Code quality

Prefer readable code over clever code.

Avoid premature abstractions.

Use type hints where they improve clarity.

Add docstrings where they provide useful information.

Important data-processing code must be tested.

Avoid putting the entire project in notebooks.

## Visualization

Visual presentation is an important part of this portfolio.

Figures should:

- have meaningful titles,
- have labeled axes,
- have appropriate legends,
- avoid unnecessary clutter,
- communicate a specific insight.

Save useful figures under `reports/figures/` so they can later be included in the README.

## README

The README will evolve during the project.

Eventually it should explain:

- the problem
- dataset
- methodology
- architecture
- experiments
- results
- visualizations
- lessons learned
- limitations
- reproduction instructions

Do not fabricate results before experiments are actually performed.

## Most important constraint

Never hide complexity from me.

If a library performs an important operation, explain conceptually what the library is doing.

The purpose of using an AI coding agent here is to accelerate implementation while I learn the underlying machine-learning and engineering concepts.

Wait for me to explicitly request each milestone before implementing the next one.