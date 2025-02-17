import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='../data/vocab_train_corpus.txt', 
    model_prefix='../out/spm_bbpe_model', 
    vocab_size=10000, 
    character_coverage=0.9995, 
    model_type='bbpe',
    num_threads=32,
    split_digits=True,
    byte_fallback=True,
    max_sentence_length=24000
)