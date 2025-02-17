from transformers import AutoTokenizer, AutoModel
import sentencepiece as spm

# set path
qwen_tokenizer_dir = "../model/qwen2.5-7b-instruct" 
chinese_sp_model_file ="../out/spm_bbpe_model.model" 

# load tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_dir)
qwen_model = AutoModel.from_pretrained(qwen_tokenizer_dir)
qwen_vocab = qwen_tokenizer.get_vocab()
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)
need_add_tokens = [chinese_sp_model.id_to_piece(i) for i in range(chinese_sp_model.get_piece_size())]
print(need_add_tokens)

# add tokens to qwen vocab
existing_tokens = set(qwen_vocab.keys())
need_add_tokens = [token for token in need_add_tokens if token not in existing_tokens]
num_added = qwen_tokenizer.add_tokens(need_add_tokens)
qwen_model.resize_token_embeddings(len(qwen_tokenizer))
print(f"Added {num_added} new tokens.")