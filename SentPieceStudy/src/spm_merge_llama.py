import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

# set path
llama_tokenizer_dir = "/path/llama-2-7b-hf" 
chinese_sp_model_file ="/path/bpe_test.model" 

# load tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)
llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

print(len(llama_tokenizer),len(chinese_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

# # print result
# 32000 10000
# ['<s>', '</s>', '<unk>']
# [1, 2, 0]
# {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}

# add chinese spm tokens to llama spm
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before: {len(llama_spm_tokens_set)}")
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

# # print result
# 32000
# Before:32000
# New model pieces: 40114 (new token: 8114)

# save merge model
output_sp_dir = '../out/merged_tokenizer_sp_test'
output_hf_dir = '../out/merged_tokenizer_hf_test'
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/chinese_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/chinese_llama.model')
tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

# 看一下效果
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
text = "The excellence of a translation can only be judged by noting"
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized length by LLaMA tokenizer:{len(llama_tokenizer.tokenize(text))}")
print(f"Tokenized by chinese_llama tokenizer:{chinese_llama_tokenizer.tokenize(text)}")
print(f"Tokenized length by LLaMA-extent-1 tokenizer:{len(chinese_llama_tokenizer.tokenize(text))}")

# # print result
# Test text:
#  The excellence of a translation can only be judged by noting
# Tokenized by LLaMA tokenizer:['▁The', '▁excell', 'ence', '▁of', '▁a', '▁translation', '▁can', '▁only', '▁be', '▁jud', 'ged', '▁by', '▁not', 'ing']
# Tokenized length by LLaMA tokenizer:14
# Tokenized by chinese_llama tokenizer:['▁The', '▁excell', 'ence', '▁of', '▁a', '▁translation', '▁can', '▁only', '▁be', '▁jud', 'ged', '▁by', '▁not', 'ing']
# Tokenized length by chinese_llama tokenizer:14

text = "麒麟，是中国古代神话中的一种瑞兽"
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized length by LLaMA tokenizer:{len(llama_tokenizer.tokenize(text))}")
print(f"Tokenized by chinese_llama tokenizer:{chinese_llama_tokenizer.tokenize(text)}")
print(f"Tokenized length by chinese_llama tokenizer:{len(chinese_llama_tokenizer.tokenize(text))}")

# # print result
# Test text:
#  麒麟，是中国古代神话中的一种瑞兽
# Tokenized by LLaMA tokenizer:['▁', '<0xE9>', '<0xBA>', '<0x92>', '<0xE9>', '<0xBA>', '<0x9F>', '，', '是', '中', '国', '古', '代', '神', '话', '中', '的', '一', '种', '<0xE7>', '<0x91>', '<0x9E>', '<0xE5>', '<0x85>', '<0xBD>']
# Tokenized length by LLaMA tokenizer:25
# Tokenized by chinese_llama tokenizer:['▁', '麒', '麟', '，', '是中国', '古代', '神', '话', '中的', '一种', '瑞', '兽']
# Tokenized length by chinese_llama tokenizer:12