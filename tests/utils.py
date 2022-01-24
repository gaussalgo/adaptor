from adaptor.utils import AdaptationArguments, StoppingStrategy

paths = {"texts": {
            "target_domain": {
                "ner": "mock_data/ner_texts_sup.txt",
                "translation": "mock_data/seq2seq_sources.txt",
                "unsup": "mock_data/domain_unsup.txt"},
            "source_domain": {}
        },
        "labels": {
            "target_domain": {
                "ner": "mock_data/ner_texts_sup_labels.txt",
                "translation": "mock_data/seq2seq_targets.txt"
            }
        }
}

test_base_models = {"translation": "Helsinki-NLP/opus-mt-cs-en",
                    "token_classification": "bert-base-multilingual-cased"}

training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=2,
                                         log_level="critical",
                                         logging_steps=1,
                                         num_train_epochs=2)
