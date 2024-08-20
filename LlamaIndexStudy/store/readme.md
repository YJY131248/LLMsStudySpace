共包含五个json文件，其中default__vector_store.json和docstore.json文件太大无法上传
请在src.chat文件中设置：

、、、
index = get_vector_index(nodes=nodes, store_path=store_path, use_store=False)
、、、

将会自动保存至当前目录
