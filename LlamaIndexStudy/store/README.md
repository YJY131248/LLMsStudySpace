共包含五个json文件，其中default__vector_store.json和docstore.json文件太大无法上传
请在src.chat.py文件的main函数中设置get_vector_index的use_store为False：

```python
index = get_vector_index(nodes=nodes, store_path=store_path, use_store=False)
```
vector-index将会自动保存至当前目录
