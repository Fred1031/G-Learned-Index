# G-Learned-Index

We implement the G-Learned Index.

These are some of the prerequisites.

```shell
nvcc compiler
c++17
```

You can test the performance of the G-Learned Index by the following at the current directory.

```shell
cd src
nvcc g-learned.cu -std=c++17 -O3 -I../include -Xcompiler -fopenmp -o g-learned
./g-learned
```

For the datasets, you can look into the data folder.

With the code and datasets, you can reproduce all the experiment results in our article. 
