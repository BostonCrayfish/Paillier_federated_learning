# Paillier_federated_learning

## Environment
- torch==0.4.0
- torchvision==0.2.0
- progress
- tqdm

## Prepare
- dataset(mnist)   
  - 利用data_prepare.py生成数据(默认mnist,还没写其它)
    - data/mnist/all_train_jpg   - - 生成的所有训练集的图片
    - data/mnist/all_test_jpg   - - 生成的所有测试集的图片
    - data/mnist/part_train_jpg   - - 生成的分成4组的训练集图片  
  - 然后将models/Trainer.py中的路径和训练集、测试集路径对应(默认mnist,还没写其它)  
    - python3 data_send.py all  可以将文件夹内所有内容都复制到其它服务器
  （要在hostlist里，而且要免密）
    - 还没写partdata对应分发的代码
- environment
  ```python
  python3 package_install.py  
    ```

## Cookbook
- run locally

    To do : 添加其它参数选项, 包括：数据集选择、文件夹选择等
```python
## 参数是训练的epoch数目
python3 -m models.Trainer 10

```

- run distributly  
  
    To do : 添加其它参数选项，包括：数据集选择、文件夹选择等
```python
## if some worker is on work , you should run python3 stop_all.py first.
python3 start_all.py
```

## Utils for debug
- Synchronize code :
```python
# send codes to other worker in hostlist
python3 data_send.py code
# send all things to other worker in hostlist
python3 data_send.py all 

```