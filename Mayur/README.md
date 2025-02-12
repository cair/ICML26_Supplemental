<!--toc:start-->
- [Experiments](#experiments)
  - [Noisy XOR example](#noisy-xor-example)
    - [Params](#params)
    - [Interpretability example](#interpretability-example)
  - [MNIST](#mnist)
    - [Params:](#params)
    - [Interpretability examples](#interpretability-examples)
      - [Ignoring partially matching symbols.](#ignoring-partially-matching-symbols)
      - [Considering partially matching symbols.](#considering-partially-matching-symbols)
  - [Fashion MNIST](#fashion-mnist)
    - [Params](#params)
    - [Params](#params)
    - [Printing clauses](#printing-clauses)
  - [Multivalue NoisyXOR:](#multivalue-noisyxor)
    - [Number of values = 4](#number-of-values-4)
      - [N0 = 3 and N1 = 0](#n0-3-and-n1-0)
      - [N0 = 3 and N1 = 1](#n0-3-and-n1-1)
      - [N0 = 3 and N1 = 2](#n0-3-and-n1-2)
      - [N0 = 3 and N1 = 3](#n0-3-and-n1-3)
  - [Questions:](#questions)
- [Multivalue NoisyXOR:](#multivalue-noisyxor)
<!--toc:end-->
# Experiments

### Noisy XOR example

![[NoisyXOR_graph.excalidraw]]

#### Params

| clauses | T   | s   | HV size | Msg size | depth |
| ------- | --- | --- | ------- | -------- | ----- |
| 4       | 100 | s   | 4       | 8        | 2     |

---

> [!INFO] Both symbols recieved same Hypervectors -> Not ideal
>
> ```c
> ⬢ [Docker] ❯ python examples/NoisyXORDemo.py
> Creating training data
> Creating testing data
> Initialization of sparse structure.
> 0 50.38 49.65 3.64 0.82
> 1 49.62 50.35 1.21 0.82
> 2 49.62 50.35 1.20 0.81
> 3 49.62 50.35 1.21 0.82
> 4 49.62 50.35 1.20 0.81
> 5 50.38 49.65 1.20 0.81
> 6 49.62 50.35 1.20 0.81
> 7 49.62 50.35 1.20 0.82
> 8 49.62 50.35 1.20 0.82
> 9 49.62 50.35 1.20 0.82
> Clause #0 W:(0 -1) x2 AND x3
> Clause #1 W:(-12 0) x2 AND NOT x0
> Clause #2 W:(7 -1)
> Clause #3 W:(2 -4) x2 AND NOT x1
> graphs_train.hypervectors=array([[3, 2],
>     [3, 2]], dtype=uint32)
> graphs_train.X.shape=(20000, 1)
> Feature literals
> Clause 0 [   0   -1]: 0 0 1 1 0 0 0 0
> Clause 1 [ -12    0]: 0 0 1 0 1 0 0 0
> Clause 2 [   7   -1]: 0 0 0 0 0 0 0 0
> Clause 3 [   2   -4]: 0 0 1 0 0 1 0 0
> 
> Message literals
> Clause 0 [   0   -1]: 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0
> Clause 1 [ -12    0]: 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0
> Clause 2 [   7   -1]: 1 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0
> Clause 3 [   2   -4]: 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0
> Actual clauses:
> Clause 0 [   0   -1]: 0 1
> Clause 1 [ -12    0]:
> Clause 2 [   7   -1]:
> Clause 3 [   2   -4]:
> Actual Messages for edge_type=0:
> Message 0 :
> Message 1 :
> Message 2 : C:1( )
> Message 3 :
> ```

The learned hyper-clauses do not match the symbols (completely).

```c
GraphTsetlinMachine on  mydev [!+?] is 📦 v0.3.1.5 via 🐍 v3.12.7 via 🅒 cotm took 33s
⬢ [Docker] ❯ python examples/NoisyXORDemo.py
Creating training data
Creating testing data
Initialization of sparse structure.
0 98.87 100.00 1.55 0.81
1 98.87 100.00 1.17 0.81
2 98.87 100.00 1.17 0.81
3 98.87 100.00 1.17 0.81
4 98.87 100.00 1.17 0.81
5 98.87 100.00 1.17 0.81
6 49.15 50.07 1.18 0.81
7 98.87 100.00 1.17 0.81
8 98.87 100.00 1.17 0.81
9 98.87 100.00 1.17 0.81
Clause #0 W:(59 -63) NOT x0
Clause #1 W:(-9 17) x0
Clause #2 W:(-36 29) x0 AND NOT x3
Clause #3 W:(-71 78) x0 AND NOT x3
graphs_train.hypervectors=array([[2, 0],
    [3, 2]], dtype=uint32)
graphs_train.X.shape=(20000, 1)
Feature literals
Clause 0 [  59  -63]: 0 0 0 0 1 0 0 0
Clause 1 [  -9   17]: 1 0 0 0 0 0 0 0
Clause 2 [ -36   29]: 1 0 0 0 0 0 0 1
Clause 3 [ -71   78]: 1 0 0 0 0 0 0 1

Message literals
Clause 0 [  59  -63]: 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0
Clause 1 [  -9   17]: 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0
Clause 2 [ -36   29]: 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0
Clause 3 [ -71   78]: 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0
Actual clauses:
Clause 0 [  59  -63]:
Clause 1 [  -9   17]:
Clause 2 [ -36   29]:
Clause 3 [ -71   78]:
Actual Messages for edge_type=0:
Message 0 :
Message 1 :
Message 2 : ~C:3( )
Message 3 : ~C:3( )
```

---

#### Interpretability example

```c
GraphTsetlinMachine on  mydev [!+?] is 📦 v0.3.1.5 via 🐍 v3.12.7 via 🅒 cotm took 50m48s
⬢ [Docker] ❯ python examples/NoisyXORDemo.py
Creating training data
Creating testing data
Initialization of sparse structure.
0 99.02 100.00 4.46 0.82
1 99.02 100.00 1.18 0.82
2 99.02 100.00 1.18 0.82
3 99.02 100.00 1.18 0.82
4 99.02 100.00 1.18 0.82
5 99.02 100.00 1.18 0.82
6 99.02 100.00 1.18 0.82
7 99.02 100.00 1.18 0.82
8 99.02 100.00 1.18 0.82
9 99.02 100.00 1.18 0.82
Clause #0 W:(-79 23) x2 AND x3 AND NOT x1 AND NOT x4 AND NOT x7
Clause #1 W:(-53 82) x2 AND x3 AND NOT x1 AND NOT x4 AND NOT x5 AND NOT x7
Clause #2 W:(4 23) x7 AND NOT x2 AND NOT x5 AND NOT x6
Clause #3 W:(96 -94) x3 AND NOT x0 AND NOT x1 AND NOT x4 AND NOT x5
Clause #4 W:(-47 41) x3 AND x7 AND NOT x0 AND NOT x1 AND NOT x2 AND NOT x6
Clause #5 W:(-20 24) NOT x0 AND NOT x4 AND NOT x5 AND NOT x6 AND NOT x7
graphs_train.hypervectors=array([[3, 2],
       [3, 7]], dtype=uint32)
tm.hypervectors=array([[14,  8],
       [ 1,  8],
       [ 2,  7],
       [15,  3],
       [ 3,  6],
       [ 2,  1]], dtype=uint32)
Feature literals
Clause 0 [ -79   23]: 0 0 1 1 0 0 0 0 0 1 0 0 1 0 0 1
Clause 1 [ -53   82]: 0 0 1 1 0 0 0 0 0 1 0 0 1 1 0 1
Clause 2 [   4   23]: 0 0 0 0 0 0 0 1 0 0 1 0 0 1 1 0
Clause 3 [  96  -94]: 0 0 0 1 0 0 0 0 1 1 0 0 1 1 0 0
Clause 4 [ -47   41]: 0 0 0 1 0 0 0 1 1 1 1 0 0 0 1 0
Clause 5 [ -20   24]: 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 1

Message literals
Clause 0 [ -79   23]: 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 0 0 0
Clause 1 [ -53   82]: 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0
Clause 2 [   4   23]: 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 1 0 1 0 0
Clause 3 [  96  -94]: 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 1 1 1 1 1 0 0
Clause 4 [ -47   41]: 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0
Clause 5 [ -20   24]: 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 1 0 1 1 1 1 0
Actual clauses:
Clause 0 [ -79   23]: 0 # A
Clause 1 [ -53   82]: 0 # A
Clause 2 [   4   23]:   # 0.5B ~0.5A
Clause 3 [  96  -94]:   # 0.5A 0.5B 
Clause 4 [ -47   41]: 1 # B
Clause 5 [ -20   24]:   # ~0.5B
Actual Messages for edge_type=0:
Message 0 : ~C:1( 0 ) C:2( ) C:4( 1 )
Message 1 : ~C:0( 0 ) C:2( ) C:3( )
Message 2 : C:3( ) C:5( )
Message 3 : 
Message 4 : C:0( 0 ) C:1( 0 )
Message 5 : C:3( ) C:4( 1 )
```

---
One of the main advantages of using tsetlin machines, is that they are inherently interpretable. Unlike the black boxes that neural networks are, it is relatively easier to examine the model and reason for the output of the model. In the case of GTM, the symbols for each node are initially encoded in vector symbolic representations. Each symbol in the graph is assigned a binary array of size HV_size, in which HV_bit bits are randomly flipped. Any combination of these symbols can be obtained by bundling(OR-ing) the hypervectors. This also means that it is possible to extract the original symbols from the hypervectors. Therefore, this extra layer of encoding does not affect the interpretability of GTM. By taking OR of a symbol hypervector and the learned clause, it is possible to check of the symbols is included in the clause.

The following example explains the clauses learned for the NoisyXOR example. The GTM is trained with 6 clauses, which are shown in the following table. Consider the following input where, N0 = B and N1 = A.
Since each node(N0 and N1) is connected to each other, they will recieve messages from each other. Thus at each Node the final clause is the AND of the activated clauses and the messages from the neighbour node. This is summarized in the Table.

The clauses activated at each node are:
N0: C2 and C3 and C4.
N1: C0 and C1 and C3 and C5.

Messages recieved at each node:
N0:  M0 and M1 and M3 and M5 = (Not C1 and C2 and C4) and (Not C0 and C2 and C3) and () and (C3 and C4). { This means that the these caluses are active at the Node N0}
N1:  M2 and M3 and M4 = (C3 and C5) and () and (C0 and C1). { This means that the these caluses are active at the Node N1}

The final clause at each node:
N0: $$
 N0 = (C2 ∩ C3 ∩ C4) ∩ ((¬C1 ∩ C2 ∩ C4) ∩ (¬C0 ∩ C2 ∩ C3) ∩ () ∩ (C3 ∩ C4))
    = C2 ∩ C3 ∩ C4 ∩ ¬C0 ∩ ¬C1
    = B ∩ ¬A ∩ A ∩ B ∩ B ∩ ¬A ∩ ¬A
 = B ∩ ¬A
$$
N1:
$$
N1 = (C0 ∩ C1 ∩ C3 ∩ C5) ∩ ((C3 ∩ C5) ∩ () ∩ (C0 ∩ C1))
    = A ∩ A ∩ A ∩ B ∩ ¬B ∩ (A ∩ B ∩ ¬B ∩ () ∩ (A ∩ A))
    = A ∩ ¬B
$$

Thus these sets of clauses are active

Thus at Node
1. X = [B,A], Y = 1
 ```c
 ipdb> X_test[0]
 array(['B', 'A'], dtype='<U1')
 ipdb> Y_test[0]
 1
 ipdb> class_sums[0]
 array([-99,  99], dtype=int32)
 ipdb> clause_outputs[0]
 array([[0, 1], # 0
        [0, 1], # 1
        [1, 0], # 2
        [1, 1], # 3
        [1, 0], # 4
        [0, 1]])# 5  
 ```
 - At Node0: `C2, C3, C4` are activated and receives messages from Node1.
  - `0.5B` and `~0.5A` and `0.5A` and `0.5B` and `B` ==> `2B` at node 0.
  - `C3` and `C5` and `C0` and `C1` = `0.5A` and `0.5B` and `~0.5B` and `A` and `A` ==> `2.5A` at node 1

 - At Node1: `C0, C1, C3, C5` are activated and receives messages from Node0.
  - `A` and `A` and `0.5A` and `0.5B` and `~0.5B` ==> `2.5A` at node 1.
  - `~C1` and `C2` and `C4` and `~C0`and `C2`and `C3` and `C3`and `C4` = `~C1` and `C2` and `C4` and `~C0` and `C3` = `~A` and `0.5B` and `~0.5A` and `B` and `~A` and `0.5A`and `0.5B` ==> `~2A`and `2B` at node 0

![[NoisyXOR-explain-BandA.excalidraw|800]]

---

2. Lets say that the input example is, X = [ A, B ], Y = 1,
 ```c
 ipdb> X_test[1]
 array(['A', 'B'], dtype='<U1')
 ipdb> Y_test[1]
 1
 ipdb> class_sums[1]
 array([-99,  99], dtype=int32)
 ipdb> clause_outputs[1]
 array([[1, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 1],
        [1, 0]], dtype=int32)
 ```

 - At Node0: `C0, C1, C3, C5` are activated and receives messages from Node1.
  - `A` and `A` and `0.5A` and `0.5B` and `~0.5B` ==> `2.5A` at node 1.
  - `~C1` and `C2` and `C4` and `~C0`and `C2`and `C3` and `C3`and `C4` = `~C1` and `C2` and `C4` and `~C0` and `C3` = `~A` and `0.5B` and `~0.5A` and `B` and `~A` and `0.5A`and `0.5B` ==> `~2A`and `2B` at node 0

 - At Node1: `C2, C3, C4` are activated and receives messages from Node0.
  - `0.5B` and `~0.5A` and `0.5A` and `0.5B` and `B` ==> `2B` at node 0.
  - `C3` and `C5` and `C0` and `C1` = `0.5A` and `0.5B` and `~0.5B` and `A` and `A` ==> `2.5A` at node 1
---

3. Lets say that the input example is, X = [ A, A ], Y = 0,
 ```c
 ipdb> X_test[2]
 array(['A', 'A'], dtype='<U1')
 ipdb> Y_test[2]
 0
 ipdb> class_sums[2]
 array([ 96, -94], dtype=int32)
 ipdb> clause_outputs[2]
 array([[0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0]], dtype=int32)
 ```

 - At Node0: `C3`is activated.
  - `0.5A`and `0.5B`

 - At Node1: `C3`is activated.
  - `0.5A`and `0.5B`
---

4. Lets say that the input example is, X = [ B,B ], Y = 0,
 ```c
 ipdb> X_test[5]
 array(['B', 'B'], dtype='<U1')
 ipdb> class_sums[5]
 array([ 96, -94], dtype=int32)
 ipdb> clause_outputs[5]
 array([[0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0]], dtype=int32)
 ```

 - At Node0: `C3`is activated.
  - `0.5A`and `0.5B`

 - At Node1: `C3`is activated.
  - `0.5A`and `0.5B`
---

| Input  | B                | A               |                                           |                                     |
| ------ | ---------------- | --------------- | ----------------------------------------- | ----------------------------------- |
| Clause | Node 0           | Node 1          | Msg from node 1 -> node 0                 | Msg from node 0 -> node 1           |
| 0      | -                | `A`             | `~A and 0.5B and ~0.5A and B`             | -                                   |
| 1      | -                | `A`             | `~A and 0.5B and ~0.5A and 0.5A and 0.5B` | -                                   |
| 2      | `0.5B and ~0.5A` | -               | -                                         | `C:3(0.5A and 0.5B) and C:5(~0.5B)` |
| 3      | `0.5A and 0.5B`  | `0.5A and 0.5B` | -                                         | -                                   |
| 4      | `B`              | -               | -                                         | `A and A`                           |
| 5      | -                | `~0.5B`         | `0.5A and 0.5B and B`                     | -                                   |
|        | `2B`             | `2.5A`          | `~3A and A and 4B`                        | `2.5A`                              |

| Input  | A               | A               |                 |                 |
| ------ | --------------- | --------------- | --------------- | --------------- |
| Input  | B               | B               |                 |                 |
| Clause | Node 0          | Node 1          | Msg from node 0 | Msg from node 1 |
| 0      | -               | -               | -               | -               |
| 1      | -               | -               | -               | -               |
| 2      | -               | -               | -               | -               |
| 3      | `0.5A and 0.5B` | `0.5A and 0.5B` | -               | -               |
| 4      | -               | -               | -               | -               |
| 5      | -               | -               | -               | -               |
|        |                 |                 |                 |                 |

---
**NOTE**: The clause `C3`(partially matching `A`and `B`) is activated no matter the input. This is similar to behavior of *empty clauses*, which also activate for any input.

---

### MNIST

Training MNIST dataset with no edges. The images are divided in 361 patches of size 10x10, which are then encoded as a single node. Therefore, each node consists of 19 x-position symbols, 19 y-position symbols, and 100 patch pixel symbols, with no edge connections.

![[MNIST-no-edges-graph.excalidraw|500]]

---
#### Params:

| clauses | T    | s   | HV size | Msg size | depth |
| ------- | ---- | --- | ------- | -------- | ----- |
| 2500    | 3125 | 10  | 128     | 256      | 1     |

```c
GraphTsetlinMachine on  mydev [!+?] is 📦 v0.3.1.5 via 🐍 v3.12.7 via 🅒 cotm took 13m7s
⬢ [Docker] ❯ python examples/MNISTConvolutionDemo.py
Training data produced
Testing data produced
Initialization of sparse structure.
0 96.65 96.87 8.51 0.63
1 97.55 97.49 6.88 0.62
2 97.95 97.88 6.65 0.62
3 98.18 98.12 6.51 0.67
4 98.38 98.10 6.44 0.66
```

---
#### Interpretability examples

 ##### Ignoring partially matching symbols.

1. Class 7
```c
Y_test[e]=7
class_sums[e]=array([-6292, -5572, -4333, -4957, -7612, -7242, -7243,  8615, -8490,
       -3873], dtype=int32)
pred=7
```

![[mnist_test_7_ignore_partial.png]]

---
2. Class 2
```c
Y_test[e]=2
class_sums[e]=array([-6858, -4546,  4568, -7485, -6907, -5848, -4708, -5079, -6663,
       -7746], dtype=int32)
pred=2
```

![[mnist_test_2_ignore_partial.png]]

---
3. Class 3
```c
Y_test[e]=1
class_sums[e]=array([-4635,  4728, -6266, -6658, -4759, -5275, -3577, -3546, -5691,
       -5794], dtype=int32)
pred=1
```

![[mnist_test_1__ignore_partial.png]]

---

 ##### Considering partially matching symbols.

![[mnist_test_7_par.png]]

---
![[mnist_test_2_par.png]]

---
![[mnist_test_1_par.png]]

---
### Fashion MNIST

#### Params

| clauses | T     | s   | HV size | Msg size | depth |
| ------- | ----- | --- | ------- | -------- | ----- |
| 8000    | 10000 | 15  | 128     | -        | 1     |

```c
GraphTM-exp on  master via 🐍 v3.12.7 via 🅒 cotm took 3s
⬢ [Docker] ❯ python /home/mayurks/work/UiA/GraphTM-exp/FMNISTConvolutionDemo.py
100%|██████████████████████████████████████████████████| 60000/60000 [13:34<00:00, 73.67it/s]
Training data produced
100%|██████████████████████████████████████████████████| 10000/10000 [02:14<00:00, 74.14it/s]
Testing data produced
Initialization of sparse structure.
0 80.77 79.62 13.09 1.10
1 83.38 82.69 11.11 1.16
2 84.81 83.69 10.86 1.11
3 85.53 84.42 10.71 1.15
4 86.36 84.95 10.62 1.10
5 86.83 85.62 10.54 1.16
6 87.18 85.68 10.61 1.12
7 87.67 85.78 10.47 1.15
8 88.05 86.22 10.42 1.16
9 88.44 86.59 10.41 1.09
10 88.83 86.84 10.35 1.17
11 88.77 86.65 10.34 1.10
12 89.20 86.79 10.30 1.16
13 89.44 86.93 10.27 1.16
14 89.47 86.94 10.31 1.10
15 89.64 86.91 10.25 1.15
16 89.75 87.14 10.24 1.10
17 89.85 87.12 10.22 1.16
18 90.08 87.20 10.20 1.10
19 90.17 87.20 10.19 1.16
20 90.43 87.20 10.18 1.16
21 90.44 87.31 10.20 1.11
22 90.57 87.50 10.15 1.22
23 90.55 87.36 10.14 1.09
24 90.85 87.60 10.12 1.15
25 90.74 87.45 10.11 1.11
26 91.08 87.57 10.09 1.15
27 90.99 87.46 10.09 1.15
28 91.20 87.50 10.16 1.11
29 91.33 87.77 10.08 1.16
30 91.42 87.50 10.08 1.09
31 91.47 87.42 10.05 1.15
32 91.47 87.56 10.05 1.11
33 91.55 87.61 10.04 1.15
34 91.74 87.73 10.04 1.18
35 91.73 87.68 10.04 1.09
36 91.67 87.84 10.01 1.15
37 91.90 87.99 10.02 1.11
38 91.79 87.95 10.00 1.15
39 92.20 87.98 10.00 1.13
40 91.85 87.98 9.99 1.15
41 92.28 88.05 9.99 1.15
42 92.11 87.82 9.99 1.09
43 92.27 87.74 9.97 1.15
44 92.28 87.96 9.98 1.09
45 92.31 87.94 9.96 1.17
46 92.36 87.78 9.96 1.09
47 92.53 87.79 9.95 1.16
48 92.42 87.98 9.94 1.17
49 92.61 87.90 9.93 1.09
```

#### Params

| clauses | T     | s   | HV size | Msg size | depth |
| ------- | ----- | --- | ------- | -------- | ----- |
| 40000   | 10000 | 15  | 128     | -        | 1     |
```c
GraphTM-exp on  master [✘!?] via 🐍 v3.12.7 via 🅒 cotm
⬢ [Docker] ❯ python /home/mayurks/work/UiA/GraphTM-exp/FMNISTConvolutionDemo.py
Training data produced
Testing data produced
Initialization of sparse structure.
0 85.89 84.57 26.78 2.94
1 88.39 86.23 23.64 2.94
2 89.62 87.20 23.31 3.02
3 90.65 87.88 23.19 2.94
4 91.26 88.30 22.98 2.94
5 91.85 88.34 22.87 2.94
6 92.30 88.47 22.81 2.95
7 92.74 88.54 22.72 2.94
8 93.39 89.07 22.65 2.94
9 93.63 88.88 22.59 2.94
10 94.00 88.93 22.54 2.94
11 94.20 89.02 22.50 2.94
12 94.45 89.07 22.50 2.94
13 94.77 89.08 22.42 2.94
14 95.17 89.23 22.39 2.95
15 95.28 89.24 22.35 2.96
16 95.36 89.02 22.32 2.94
17 95.62 89.08 22.31 2.95
18 96.07 89.35 22.26 2.94
19 96.16 89.22 22.24 2.94
20 96.33 89.37 22.22 2.94
21 96.51 89.25 22.19 2.94
22 96.78 89.34 22.16 2.94
23 96.71 89.54 22.14 2.95
24 96.85 89.23 22.12 2.94
25 97.03 89.27 22.10 2.94
26 97.15 89.33 22.08 2.94
27 97.19 89.20 22.07 2.94
28 97.47 89.30 22.06 2.95
29 97.51 89.33 22.04 2.94
30 97.73 89.64 22.02 2.94
31 97.65 89.51 22.01 2.94
32 97.77 89.33 21.97 2.94
33 97.87 89.43 21.96 2.93
34 97.90 89.30 21.99 2.94
35 97.94 89.44 22.00 2.94
36 98.15 89.64 21.92 2.94
37 98.27 89.50 21.94 2.94
38 98.31 89.47 21.90 2.94
39 98.41 89.30 21.88 2.94
40 98.41 89.12 21.87 2.94
41 98.35 89.22 21.87 2.94
42 98.56 89.20 21.98 2.95
43 98.58 89.64 21.89 2.95
44 98.62 89.18 21.82 2.94
45 98.56 89.22 21.82 2.94
46 98.65 89.31 21.82 2.94
47 98.78 89.44 21.81 3.06
48 98.80 89.51 21.81 2.94
49 98.89 89.53 21.79 2.94
```

#### Printing clauses
```c
Y_test[e]=9
class_sums[e]=array([-27001, -39915, -35587, -34831, -26786, -14819, -31023, -14118,
       -24254,  17116], dtype=int32)
pred=9
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40000/40000 [00:11<00:00, 3513.18it/s]
```
![[fmnist_test_Trouser.png]]

```c
Y_test[e]=2
class_sums[e]=array([-20686, -21771,  23114, -36740,  -5670, -29174, -29456, -26943,
       -27598, -29225], dtype=int32)
pred=2
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40000/40000 [00:14<00:00, 2831.45it/s]
```
![[fmnist_test_Pullover.png]]
```c
Y_test[e]=1
class_sums[e]=array([-15812,  35051, -28821, -23355, -27669, -19740, -31123, -30566,
       -22221, -20537], dtype=int32)
pred=1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40000/40000 [00:11<00:00, 3390.75it/s]
```

![[fmnist_test_Ankle_boot.png]]

### Multivalue NoisyXOR:

#### Number of values = 4

Symbols:

| A   | B   | C   | D   |
| --- | --- | --- | --- |
| 0   | 1   | 2   | 3   |

Dataset:
- Y = 1, if both N0 and N1 are even
- Y = 1, if both N0 and N1 are odd
- Y = 0, Otherwise

Params:

| clauses | T   | s   | HV size | Msg size | depth |
| ------- | --- | --- | ------- | -------- | ----- |
| 12      | 100 | 2   | 8       | 16       | 2     |

```c
GraphTM-exp on  master [!] via 🐍 v3.12.7 via 🅒 cotm took 20s
⬢ [Docker] ❯ python MultivalueXOR.py
Creating training data
Creating testing data
Initialization of sparse structure.
0 99.02 100.00 1.63 0.86
1 99.02 100.00 1.24 0.86
2 99.02 100.00 1.24 0.86
3 99.02 100.00 1.24 0.86
4 99.02 100.00 1.24 0.86
Clause #0 W:(85 -84) x2
Clause #1 W:(34 -30) NOT x3
Clause #2 W:(-4 5) x6
Clause #3 W:(-161 150) x0 AND NOT x1
Clause #4 W:(22 -16) NOT x2
Clause #5 W:(50 -45) x0 AND NOT x6
Clause #6 W:(25 -29) NOT x0 AND NOT x2
Clause #7 W:(-3 -2)
Clause #8 W:(-132 138) x1 AND x2 AND NOT x3 AND NOT x4 AND NOT x5 AND NOT x6 AND NOT x7
Clause #9 W:(49 -43) NOT x0 AND NOT x2
Clause #10 W:(-49 56) NOT x0
Clause #11 W:(10 -14) NOT x0 AND NOT x3
{'Plain': 0}
graphs_train.hypervectors=array([[6, 5],
       [2, 1],
       [7, 1],
       [0, 7]], dtype=uint32)
tm.hypervectors=array([[14,  0],
       [ 9, 13],
       [ 9,  2],
       [ 7,  8],
       [ 0,  4],
       [ 5, 12],
       [ 2, 15],
       [ 3, 11],
       [ 8,  3],
       [ 7,  4],
       [ 2,  5],
       [ 9,  1]], dtype=uint32)
Feature literals
Clause 0 [  85  -84]: 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 1 [  34  -30]: 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
Clause 2 [  -4    5]: 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
Clause 3 [-161  150]: 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
Clause 4 [  22  -16]: 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
Clause 5 [  50  -45]: 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
Clause 6 [  25  -29]: 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
Clause 7 [  -3   -2]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 8 [-132  138]: 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1
Clause 9 [  49  -43]: 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
Clause 10 [ -49   56]: 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
Clause 11 [  10  -14]: 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0

Message literals
Clause 0 [  85  -84]: 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 1 [  34  -30]: 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
Clause 2 [  -4    5]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 3 [-161  150]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
Clause 4 [  22  -16]: 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0
Clause 5 [  50  -45]: 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 6 [  25  -29]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
Clause 7 [  -3   -2]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
Clause 8 [-132  138]: 1 1 1 0 1 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0
Clause 9 [  49  -43]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
Clause 10 [ -49   56]: 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 11 [  10  -14]: 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Actual clauses:
Clause 0 [  85  -84]: 0.5B
Clause 1 [  34  -30]:
Clause 2 [  -4    5]: 0.5A
Clause 3 [-161  150]: ~0.5B ~0.5C 0.5D
Clause 4 [  22  -16]: ~0.5B
Clause 5 [  50  -45]: ~0.5A 0.5D
Clause 6 [  25  -29]: ~0.5B ~0.5D
Clause 7 [  -3   -2]:
Clause 8 [-132  138]: ~1.0A 1.0B 0.5C ~0.5C ~0.5D
Clause 9 [  49  -43]: ~0.5B ~0.5D
Clause 10 [ -49   56]: ~0.5D
Clause 11 [  10  -14]: ~0.5D
Actual Messages for edge_type=0:
Message 0 :
Message 1 : C:5( ~0.5A 0.5D )
Message 2 :
Message 3 :
Message 4 : C:1( )
Message 5 :
Message 6 :
Message 7 :
Message 8 : C:2( 0.5A ) C:4( ~0.5B ) C:6( ~0.5B ~0.5D ) C:9( ~0.5B ~0.5D ) C:11( ~0.5D )
Message 9 :
Message 10 :
Message 11 :
```

###### N0 = 3 and N1 = 0
```c
ipdb> X_test[0]
array([3., 0.])
ipdb> Y_test[0]
1
ipdb> class_sums[0]
array([-158,  165], dtype=int32)
ipdb> clause_outputs[0]
array([[0, 0],
       [0, 1],
       [0, 1],
       [1, 0],
       [1, 1],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 1],
       [0, 0]], dtype=int32)
```

At N0:
- Active clauses are: `C3`, `C4` = `~0.5B ~0.5C 0.5D`,  `~0.5B` = `~B, ~C, D`
- Messages received: `~0.5A 0.5D`, ` `, ` `, ` ` = `~A, D`

At N1:
- Active Clauses are: `C1`, `C2`, `C4`, `C10` = ` `, `0.5A`, `~0.5B`, `~0.5D`  = `A, ~B, ~D`
- Messages received: ` `, ` ` = ` `

###### N0 = 3 and N1 = 1
```c
ipdb> X_test[4]
array([3., 1.])
ipdb> Y_test[4]
0
ipdb> class_sums[4]
array([ 120, -103], dtype=int32)
ipdb> clause_outputs[4]
array([[0, 1],
       [0, 1],
       [0, 0],
       [0, 0],
       [0, 0],
       [1, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 1],
       [0, 0]], dtype=int32)
```

###### N0 = 3 and N1 = 2
```c
ipdb> X_test[13]
array([3., 2.])
ipdb> Y_test[13]
1
ipdb> class_sums[13]
array([-154,  160], dtype=int32)
ipdb> clause_outputs[13]
array([[0, 0],
       [0, 1],
       [0, 0],
       [1, 0],
       [1, 1],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 1],
       [0, 0]], dtype=int32)
```

###### N0 = 3 and N1 = 3
```c
ipdb> X_test[31]
array([3., 3.])
ipdb> Y_test[31]
0
ipdb> class_sums[31]
array([106, -91], dtype=int32)
ipdb> clause_outputs[31]
array([[0, 0],
       [1, 1],
       [0, 0],
       [0, 0],
       [1, 1],
       [1, 1],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0]], dtype=int32)
```

```c
⬢ [Docker] ❯ python MultivalueXOR.py
Creating training data
Creating testing data
Initialization of sparse structure.
0 98.91 100.00 1.54 0.80
1 98.91 100.00 1.16 0.80
2 98.91 100.00 1.16 0.80
3 98.91 100.00 1.16 0.80
4 98.91 100.00 1.16 0.80
Clause #0 W:(-23 34) NOT x0 AND NOT x1 AND NOT x5 AND NOT x7 AND NOT x11 AND NOT x12
Clause #1 W:(117 -121) NOT x3 AND NOT x4 AND NOT x8 AND NOT x10 AND NOT x12 AND NOT x15
Clause #2 W:(-17 18) NOT x0 AND NOT x1 AND NOT x4 AND NOT x5 AND NOT x6 AND NOT x7 AND NOT x11 AND NOT x12 AND NOT x14
Clause #3 W:(-82 77) NOT x0 AND NOT x2 AND NOT x3 AND NOT x4 AND NOT x7 AND NOT x8 AND NOT x9 AND NOT x10 AND NOT x12 AND NOT x14 AND NOT x15
Clause #4 W:(19 -27) NOT x5 AND NOT x6 AND NOT x11
Clause #5 W:(-4 4) NOT x0 AND NOT x1 AND NOT x4 AND NOT x5 AND NOT x7 AND NOT x9 AND NOT x12 AND NOT x14
Clause #6 W:(28 -29) NOT x1 AND NOT x5 AND NOT x6
Clause #7 W:(-6 4) NOT x2 AND NOT x8 AND NOT x14
Clause #8 W:(-73 66) NOT x1 AND NOT x3 AND NOT x5 AND NOT x6 AND NOT x11 AND NOT x14 AND NOT x15
Clause #9 W:(52 -42) NOT x6 AND NOT x11
Clause #10 W:(-46 39) x1 AND x5
Clause #11 W:(30 -22) NOT x0 AND NOT x2 AND NOT x3 AND NOT x4 AND NOT x9 AND NOT x10 AND NOT x14
{'Plain': 0}
graphs_train.hypervectors=array([[ 6, 13],
       [ 2, 15],
       [ 5,  1],
       [ 3, 15],
       [11,  6],
       [ 8,  9],
       [13, 11],
       [10, 13]], dtype=uint32)
tm.hypervectors=array([[12,  4],
       [ 5,  7],
       [14, 10],
       [ 5,  7],
       [11,  7],
       [ 4,  2],
       [ 1, 10],
       [ 6,  1],
       [10,  7],
       [10, 13],
       [ 7, 10],
       [ 6, 14]], dtype=uint32)
Feature literals
Clause 0 [ -23   34]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 1 1 0 0 0
Clause 1 [ 117 -121]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 0 1 0 0 1
Clause 2 [ -17   18]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 0 0 0 1 1 0 1 0
Clause 3 [ -82   77]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 1 1 1 1 0 1 0 1 1
Clause 4 [  19  -27]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0
Clause 5 [  -4    4]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 0 1 0 0 1 0 1 0
Clause 6 [  28  -29]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0
Clause 7 [  -6    4]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0
Clause 8 [ -73   66]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 1 0 0 0 0 1 0 0 1 1
Clause 9 [  52  -42]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0
Clause 10 [ -46   39]: 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 11 [  30  -22]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 1 1 0 0 0 1 0

Message literals
Clause 0 [ -23   34]: 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0
Clause 1 [ 117 -121]: 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
Clause 2 [ -17   18]: 0 0 0 0 0 1 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
Clause 3 [ -82   77]: 0 1 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 0 0 1 0 1 0 0 1 1 0 0 0 0 0 1
Clause 4 [  19  -27]: 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 5 [  -4    4]: 0 1 1 0 1 1 1 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1
Clause 6 [  28  -29]: 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 7 [  -6    4]: 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0
Clause 8 [ -73   66]: 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
Clause 9 [  52  -42]: 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 10 [ -46   39]: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Clause 11 [  30  -22]: 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1
Actual clauses:
Clause 0 [ -23   34]: ~1.0C ~0.5E ~0.5G
Clause 1 [ 117 -121]: ~0.5B ~1.0D ~0.5F ~0.5H
Clause 2 [ -17   18]: ~0.5A ~1.0C ~1.0E ~0.5G
Clause 3 [ -82   77]: ~1.0B ~1.0D ~1.0F ~0.5H
Clause 4 [  19  -27]: ~0.5A ~0.5C ~1.0E ~0.5G
Clause 5 [  -4    4]: ~1.0C ~0.5F
Clause 6 [  28  -29]: ~0.5A ~1.0C ~0.5E
Clause 7 [  -6    4]: ~0.5B ~0.5F
Clause 8 [ -73   66]: ~0.5A ~0.5B ~1.0C ~1.0D ~1.0E ~0.5G
Clause 9 [  52  -42]: ~0.5A ~1.0E ~0.5G
Clause 10 [ -46   39]: 1.0C
Clause 11 [  30  -22]: ~0.5B ~0.5D ~0.5F ~0.5H
Actual Messages for edge_type=0:
Message 0 : 0.5C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 0.5C:3( ~1.0B ~1.0D ~1.0F ~0.5H )
Message 1 : 1.0C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 1.0C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 0.5C:8( ~0.5A ~0.5B ~1.0C ~1.0D ~1.0E ~0.5G ) 0.5C:10( 1.0C )
Message 2 : 1.0C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 1.0C:2( ~0.5A ~1.0C ~1.0E ~0.5G ) 1.0C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 0.5C:6( ~0.5A ~1.0C ~0.5E ) 1.0C:8( ~0.5A ~0.5B ~1.0C ~1.0D ~1.0E ~0.5G ) 0.5C:9( ~0.5A ~1.0E ~0.5G ) 1.0C:10( 1.0C ) 0.5C:11( ~0.5B ~0.5D ~0.5F ~0.5H )
Message 3 : 0.5C:0( ~1.0C ~0.5E ~0.5G ) 0.5C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) ~0.5C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 1.0C:2( ~0.5A ~1.0C ~1.0E ~0.5G ) 0.5C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) ~0.5C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) 1.0C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 1.0C:5( ~1.0C ~0.5F ) 1.0C:6( ~0.5A ~1.0C ~0.5E ) 0.5C:7( ~0.5B ~0.5F ) 1.0C:8( ~0.5A ~0.5B ~1.0C ~1.0D ~1.0E ~0.5G ) 1.0C:9( ~0.5A ~1.0E ~0.5G ) 1.0C:10( 1.0C ) 0.5C:11( ~0.5B ~0.5D ~0.5F ~0.5H )
Message 4 : 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G )
Message 5 : 0.5C:0( ~1.0C ~0.5E ~0.5G ) 1.0C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 0.5C:2( ~0.5A ~1.0C ~1.0E ~0.5G ) 1.0C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 1.0C:5( ~1.0C ~0.5F ) 0.5C:6( ~0.5A ~1.0C ~0.5E ) 1.0C:7( ~0.5B ~0.5F ) 0.5C:8( ~0.5A ~0.5B ~1.0C ~1.0D ~1.0E ~0.5G ) 0.5C:10( 1.0C ) 1.0C:11( ~0.5B ~0.5D ~0.5F ~0.5H )
Message 6 : 0.5C:0( ~1.0C ~0.5E ~0.5G ) 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 0.5C:5( ~1.0C ~0.5F ) 0.5C:9( ~0.5A ~1.0E ~0.5G )
Message 7 : 0.5C:0( ~1.0C ~0.5E ~0.5G ) ~0.5C:0( ~1.0C ~0.5E ~0.5G ) 1.0C:5( ~1.0C ~0.5F ) 0.5C:7( ~0.5B ~0.5F ) 0.5C:11( ~0.5B ~0.5D ~0.5F ~0.5H )
Message 8 : 0.5C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 0.5C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) 0.5C:6( ~0.5A ~1.0C ~0.5E ) 0.5C:7( ~0.5B ~0.5F )
Message 9 : 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 0.5C:9( ~0.5A ~1.0E ~0.5G )
Message 10 :
Message 11 : 1.0C:1( ~0.5B ~1.0D ~0.5F ~0.5H ) 1.0C:3( ~1.0B ~1.0D ~1.0F ~0.5H ) 0.5C:4( ~0.5A ~0.5C ~1.0E ~0.5G ) 0.5C:8( ~0.5A ~0.5B ~1.0C ~1.0D ~1.0E ~0.5G ) ~0.5C:9( ~0.5A ~1.0E ~0.5G ) 0.5C:10( 1.0C )

```

