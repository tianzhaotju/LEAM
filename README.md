# LEAM

LEAM adapts the syntax-guided encoder-decoder architecture by extending a set of grammar rules specific to our mutation task, to guarantee syntactic correctness of constructed mutation faults. It predicts the statements to be mutated under the context of the targeted method to reduce search space, and then predicts grammar rules for mutation fault construction based on both semantic and structural features in AST.

See [Zhao Tian](https://tianzhaotju.github.io/), [Junjie Chen](https://sites.google.com/site/junjiechen08/), et al. "[Learning to Construct Better Mutation Faults](./LEAM.pdf)." The 37th IEEE/ACM International Conference on Automated Software Engineering (ASE'22).

 🏆 ACM SIGSOFT Distinguished Paper Award 

## 1. Environment

* Python 3.7
* PyTorch 1.3
* Defects4J V1.0
* Java 8



## 2. A Demo

Pleas find our built model at: <https://huggingface.co/anonymoussss/leam/blob/main/best_model.ckpt>. Download the model and place it in folder `LEAM/checkpointSearch/`.

We provide a demo to generate mutation faults for JFree Chart on fiexd version.

Please run `CUDA_VISIBLE_DEVICES=0,1 python3 testDefect4jV1Fixed.py Chart-1` in the `LEAM/` folder directory. The generated mutation faults are saved as JSON in the `LEAM/mutants/` folder directory.

(1) `CUDA_VISIBLE_DEVICES=0,1`: specify two available GPUs.
(2) `Chart-1`: `project name` and `bug id`.



## 3. Experimental Results

####  (1) Distribution of real faults by the number of involved statements
<img src="./pictures/number.png" alt="drawing" width="800">

--- --- ---



#### (2) Ability of representing real faults interms of adequate test suites. (a) shows the result when using all the constructed mutation faults; (b)/(c) shows the result under controlling for the number of mutation faults when considering/ignoring DeepMutation (DM)
<img src="./pictures/realfault.png" alt="drawing" width="500">

--- ---




#### (3) Ability of representing mutation faults constructed by other mutation techniques. (a) shows the result when using all the constructed mutation faults; (b)/(c) shows the result under controlling for the number of mutation faults when considering/ignoring DeepMutation (DM)
<img src="./pictures/mutationfault.png" alt="drawing" width="800">

--- ---




#### (4) Correlation between mutation score and real fault detection
<img src="./pictures/correlation.png" alt="drawing" width="500">

--- ---




#### (5) Effectiveness comparison in mutation-based TCP when ignoring/considering Closure (detailed experimental data can be found in the `experiment/tcp.csv` folder)
<center> <div><b>(a) Ignoring Closure</b><div/>
<img src="./pictures/tcp_1.png" alt="drawing" width="800">
<center/>
<br>
<center> <div><b>(b) Considering Closure<b/><div/>
<img src="./pictures/tcp_2.png" alt="drawing" width="800">
<center/>


--- ---




#### (6) Effectiveness comparison in mutation-based FL when ignoring/considering Closure  (detailed experimental data can be found in the `experiment/fl.csv` folder)
<center> <div><b>(a) Ignoring Closure<b/><div/>
<img src="./pictures/fl_1.png" alt="drawing" width="800">
<center/>
<br>
<center> <div><b>(b) Considering Closure<b/><div/>
<img src="./pictures/fl_2.png" alt="drawing" width="800">
<center/>


--- ---



#### (7) Comparison effectiveness between LEAM and its variant w/o statement prediction (SP) in mutation-based TCP (Chart, Lang, Math, Time)

<img src="./pictures/ablationtcp.png" alt="drawing" width="600">

--- ---




#### (8) Comparison effectiveness between LEAM and its variant w/o statement prediction (SP) in mutation-based FL (Chart, Lang, Math, Time)
<img src="./pictures/ablationfl.png" alt="drawing" width="600">

--- ---




#### (9) The influence of the number of mutation faults, which can be controlled by beam size, on mutation-based TCP (Chart)
<img src="./pictures/beamsizetcp.png" alt="drawing" width="800">

--- ---




#### (10) The influence of the number of mutation faults, which can be controlled by beam size, on mutation-based FL (Chart)
<img src="./pictures/beamsizetcp.png" alt="drawing" width="800">

--- ---
