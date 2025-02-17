import sentencepiece as spm

sp_bpe = spm.SentencePieceProcessor() 
sp_bpe.load('../out/spm_bbpe_model.model')

print('*** BPE ***')
print(sp_bpe.encode_as_pieces('The excellence of a translation can only be judged by noting'))
print(len(sp_bpe.encode_as_pieces('The excellence of a translation can only be judged by noting')))
print(sp_bpe.encode_as_pieces('麒麟，是中国古代神话中的一种瑞兽'))
print(len(sp_bpe.encode_as_pieces('麒麟，是中国古代神话中的一种瑞兽')))