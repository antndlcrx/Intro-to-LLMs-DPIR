<p align="center">
  <img src="images/dpir_oss.png" alt="LLMs for Social Science — DPIR Oxford" width="560">
</p>

# LLMs for Social Science (DPIR)

Teaching materials for the **Large Language Models for Social Science** workshop at the Department of Politics and International Relations (DPIR), University of Oxford.

This repository gives you a practical path from how language models work internally to deploying them in real research workflows. Each day pairs concepts with hands-on Python notebooks.

## Course overview

The workshop is structured as a five-day progression: foundations of representation and generation; turning raw models into usable tools; deployment and validation for research; social-science applications (including RAG and simulated survey responses); and agent-style workflows with a capstone project.

The authoritative schedule, core topics, readings, and recommended tools are in the syllabus PDF:

- [`syllabus_oss_2025.pdf`](syllabus_oss_2025.pdf)

## Prerequisites

- Comfortable reading and writing Python at a beginner to intermediate level (strongly recommended).
- A Google account if you run the notebooks on **Google Colab** (typical for GPU-backed exercises).

If you need to catch up on basics, see the notebooks under [`preliminaries/`](preliminaries/).

## Course notebooks

| Day | Theme | Notebook |
|-----|--------|----------|
| 1 | Foundations — from embeddings to transformers | [`course_notebooks/day_1_foundations/day1_embeddings_to_transformers.ipynb`](course_notebooks/day_1_foundations/day1_embeddings_to_transformers.ipynb) |
| 2 | From models to tools — post-training, prompting, reasoning, evaluation | [`course_notebooks/day_2_models_to_tools/day2_from_models_to_tools.ipynb`](course_notebooks/day_2_models_to_tools/day2_from_models_to_tools.ipynb) |
| 3 | Deploying for research — fine-tuning, APIs, classification and validation | [`course_notebooks/day_3_deploying_for_research/day3_deploying_for_research.ipynb`](course_notebooks/day_3_deploying_for_research/day3_deploying_for_research.ipynb) |
| 4 | Social science applications — extraction, RAG, LLMs as simulated agents | [`course_notebooks/day_4_rag_survey_response_prediciton/day4_social_science_applications.ipynb`](course_notebooks/day_4_rag_survey_response_prediciton/day4_social_science_applications.ipynb) |
| 5 | Agentic workflows — tools, planning, research assistants, capstone | [`course_notebooks/day_5_agents/day5_agentic_workflows.ipynb`](course_notebooks/day_5_agents/day5_agentic_workflows.ipynb) |

Open any `.ipynb` file on GitHub and use **“Open in Colab”** from the notebook menu, or clone this repository and run locally with Jupyter / VS Code or your preffered alternative.

## Supplementary materials

| Resource | Description |
|----------|-------------|
| [`preliminaries/quick_python_intro.ipynb`](preliminaries/quick_python_intro.ipynb) | Short Python refresher |
| [`preliminaries/quick_embedding_intro.ipynb`](preliminaries/quick_embedding_intro.ipynb) | Embeddings and basic NLP concepts |
| [`preliminaries/quick_dl_torch_intro.ipynb`](preliminaries/quick_dl_torch_intro.ipynb) | Deep learning and PyTorch essentials |
| [`data/`](data/) | Small example datasets used in some exercises |

## Course convenor

**Maksim Zubok** — Nuffield College, University of Oxford (`maksim.zubok@nuffield.ox.ac.uk`).

## Archived course materials

Earlier **Spring School / OSS** notebooks and the previous course README (including historical Colab links) are kept under [`archive/oss_2025/README.md`](archive/oss_2025/README.md). Older tutorial notebooks from 2024 are under [`archive/oss_2024/`](archive/oss_2024/).

## License

See [`LICENSE`](LICENSE).
