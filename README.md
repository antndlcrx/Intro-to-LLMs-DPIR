# LLMs for Social Science

**Workshop for social sciences**  
April 20–21, 2026 · European University Institute · Florence, Italy

Teaching materials for the **Large Language Models for Social Science** workshop.

This repository gives you a practical path from how language models work internally to deploying them in real research workflows. Each module pairs concepts with hands-on Python notebooks.

<!-- Header image (`images/dpir_oss.png`) omitted for this venue; restore for Oxford / other programmes if needed. -->

## Course overview

The full course is structured as five modules in sequence: foundations of representation and generation; turning raw models into usable tools; deployment and validation for research; social-science applications (including RAG and simulated survey responses); and agent-style workflows. Shorter workshops typically cover a subset of these modules.

The schedule, core topics, readings, and recommended tools for this symposium session are in the syllabus PDF:

- [`Workshop_Tue_7_Maksim_Zubok.pdf`](Workshop_Tue_7_Maksim_Zubok.pdf)

## Prerequisites

- Comfortable reading and writing Python at a beginner to intermediate level (strongly recommended).
- A Google account if you run the notebooks on **Google Colab** (typical for GPU-backed exercises).

If you need to catch up on basics, see the notebooks under [`preliminaries/`](preliminaries/).

## Course notebooks

These exercises are companion notebooks to the course text and visualizations on the official course page: [LLMs for Social Science course](https://llmsforsocialscience.net/course/).

| Module | Theme | Notebook |
|-----|--------|----------|
| 1 | Foundations: from embeddings to transformers | [`Module 1 notebook`](course_notebooks/day_1_foundations/day1_embeddings_to_transformers.ipynb) |
| 2 | From models to tools: post-training, prompting, reasoning, evaluation | [`Module 2 notebook`](course_notebooks/day_2_models_to_tools/day2_from_models_to_tools.ipynb) |
| 3 | Deploying for research: fine-tuning, APIs, classification and validation | [`Module 3 notebook`](course_notebooks/day_3_deploying_for_research/day3_deploying_for_research.ipynb) |
| 4 | Social science applications: extraction, RAG, LLMs as simulated agents | [`Module 4 notebook`](course_notebooks/day_4_rag_survey_response_prediciton/day4_social_science_applications.ipynb) |
| 5 | Agentic workflows: tools, planning, research assistants, capstone | [`Module 5 notebook`](course_notebooks/day_5_agents/day5_agentic_workflows.ipynb) |

Open any `.ipynb` file on GitHub and use **“Open in Colab”** from the notebook menu, or clone this repository and run locally with Jupyter / VS Code or your preferred alternative.

## Supplementary materials

| Resource | Description |
|----------|-------------|
| [`preliminaries/quick_python_intro.ipynb`](preliminaries/quick_python_intro.ipynb) | Short Python refresher |
| [`preliminaries/quick_embedding_intro.ipynb`](preliminaries/quick_embedding_intro.ipynb) | Embeddings and basic NLP concepts |
| [`preliminaries/quick_dl_torch_intro.ipynb`](preliminaries/quick_dl_torch_intro.ipynb) | Deep learning and PyTorch essentials |
| [`data/`](data/) | Small example datasets used in some exercises |

## Course convenor

**Maksim Zubok**, Nuffield College, University of Oxford (`maksim.zubok@nuffield.ox.ac.uk`).

## Archived course materials

Earlier **Spring School / OSS** notebooks and the previous course README (including historical Colab links) are kept under [`archive/oss_2025/README.md`](archive/oss_2025/README.md). Older tutorial notebooks from 2024 are under [`archive/oss_2024/`](archive/oss_2024/).

## License

See [`LICENSE`](LICENSE).
