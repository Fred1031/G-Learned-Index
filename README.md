# G-Learned-Index

These are some of the prerequisites.

```shell
nvcc compiler
c++17
```

One can test the performance of the G-Learned Index by the following at the current directory.

```shell
cd src
nvcc g-learned.cu -std=c++17 -O3 -I../include -Xcompiler -fopenmp -o g-learned
./g-learned
```

Due to limited space provided by GoogleDrive, we only upload a sample code and a relatively small input (both the dataset and the query). Full codes and datasets will be realeased on our GitHub repository as soon as possible.

ASSMJoin

We implement the 14 algorithms in ASSMJoin and other approaches for comparison here. The experimental environment is shown in the table below.

Component	Description
Processor	Intel® Core™ i9-10900X, 2(socket)*10*3.70GHz
L3 cache	19.25M
Instruction set extension	SSE4.1, SSE4.2, AVX2, AVX-512
Memory	4*32G DDR4, 2400 MHz
OS	Linux 5.11.0
Compiler	g++ -O3
For the datasets, you can fetch them at https://anonymous.4open.science/r/DatasetofASSMJoin-8089/.

To run the code, please firstly determine the experiment directory and L3 cache size in ./run_all.sh as well as the mapping of your intel cpu in ./cpu-mapping.txt. Note that the directory of dataset should match the experiment directory you set.

With the code and datasets, you can reproduce all the experiment results in our article. It takes us weeks to run them on the i9-10900X, becuase quality measures need a number of repeat experiment. If you want a quick check with less precision, you can modify the scripts and resize the group size of the quality measure. It will only take a few days to produce an initial result.
