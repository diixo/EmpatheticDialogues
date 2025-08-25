

def create_pretrain_args() -> list:
    return [
        "retrieval_train.py",
        "--batch-size", "64",
        "--cuda",
        "--dataset-name", "empchat",    # to select chat-type, require reddit_folder
        "--dict-max-words", "57000",    # 57k is size of word_dictionary.pth
        "--display-iter", "100",
        "--embeddings", "crawl-300d-2M-vec/fasttext-crawl-300d-2M.txt", # input, crawl-300d-2M-no-header.txt
        # https://worksheets.codalab.org/worksheets/0x84b71dd010cf4bff8d9f59cc22b49344
        "--empchat-folder", "empatheticdialogues",  # input dataset, splited on: train, valid, test
        "--learn-embeddings",
        "--learning-rate", "5e-4",
        "--model", "transformer",
        "--model-dir", "train-products",    # for created model
        "--model-name", "test-model",       # will create "model-name.mdl" into model-dir
        "--n-layers", "4",
        "--num-epochs", "100",
        "--optimizer", "adamax",
        "--reddit-folder", "reddit-data",   # input for: word_dictionary.pth (hardcoded)
        "--transformer-dim", "300",
        "--transformer-n-heads", "6",
        #--reactonly,
        #--max-hist-len, "4",
    ]

def create_fine_tuning_args() -> list:
    return [
        "retrieval_train.py",
        "--batch-size", "64",
        "--cuda",
        "--dataset-name", "empchat",    # to select chat-type, require reddit_folder
        "--dict-max-words", "57000",    # 57k is size of word_dictionary.pth
        "--display-iter", "100",
        "--load-checkpoint", "pretrain-products/test-model.mdl",   # input filepath
        "--empchat-folder", "empatheticdialogues",  # input dataset, splited on: train, valid, test
        "--learn-embeddings",
        "--learning-rate", "5e-4",
        "--model", "transformer",
        "--model-dir", "train-products",    # for created model
        "--model-name", "test-model",       # will create "model-name.mdl" into model-dir
        "--n-layers", "4",
        "--num-epochs", "10",
        "--optimizer", "adamax",
        "--reddit-folder", "reddit-data",   # input for: word_dictionary.pth (hardcoded)
        "--transformer-dim", "300",
        "--transformer-n-heads", "6",
        #--reactonly,
        #--max-hist-len, "4",
    ]


def create_evaluation_args() -> list:
    return [
        "retrieval_train.py",
        "--batch-size", "64",
        "--cuda",
        "--dataset-name", "empchat",    # to select chat-type, require reddit_folder
        "--dict-max-words", "250000",    # 57k is size of word_dictionary.pth
        "--display-iter", "100",
        "--empchat-folder", "empatheticdialogues",  # input dataset, splited on: train, valid, test
        "--max-hist-len", "4",
        "--model", "transformer",
        "--model-dir", "train-products",    # for created model
        #"--model-name", "test-model",       # will create "model-name.mdl" into model-dir
        "--n-layers", "4",
        "--num-epochs", "100",
        "--optimizer", "adamax",
        "--pretrained", "models/normal_transformer_finetuned.mdl",   # input for: word_dictionary.pth (hardcoded)
        "--transformer-dim", "300",
        "--transformer-n-heads", "6",
        #--reactonly,
        #--max-hist-len, "4",
    ]

