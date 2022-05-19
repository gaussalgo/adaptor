## Tutorials

This section contains demonstrations of Adaptor's applications for various use-cases.

### What can you find in each tutorial

* [adapted_named_entity_recognition](adapted_named_entity_recognition.ipynb) constructs a standard **single-objective** token classification training pipeline and compares it to the common **multi-objective** approach applying unsupervised **adaptation** (Masked Language Modeling objective) preceding the final objective (Token Classification objective). It also shows how to ease **evaluation** of the persisted model **using objectives**.
* [robust_classification](robust_classification.ipynb) **trains the NLI** (Natural Language Inference) **classifier** on MNLI and **evaluates** the model on a held-out adversarial objective (HANS) during the training. We assess whether the **model selection** based on in-domain accuracy (MNLI) also **brings the most-robust model** and **what in-domain price we need to pay** to make the model robust to other dataset (HANS).
* [unsupervised_machine_translation](unsupervised_machine_translation.ipynb) uses Adaptor to assess the gap between the supervised and a commonly-used unsupervised objective (BackTranslation).