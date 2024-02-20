<p align="left"> <img src="https://raw.githubusercontent.com/fajri91/eval_picts/master/ArabicMMLU-Bar.png" style="width: 100%;" id="title-icon">  </p>
<p align="left"> <i>Fajri Koto, Haonan Li, Sara Shatanawi, Jad Doughman, Abdelrahman Boda Sadallah, Aisha Alraeesi, Khalid Almubarak, Zaid Alyafeai, Neha Sengupta, Shady Shehata, Nizar Habash, Preslav Nakov, and Timothy Baldwin </i></p>

<h4 align="left">
    MBZUAI, Prince Sattam bin Abdulaziz University, KFUPM, Core42, NYU Abu Dhabi, The University of Melbourne
</h4>

## Introduction

We present ArabicMMLU, the first multi-task language understanding benchmark for Arabic language, sourced from school exams across diverse educational levels in different countries spanning North Africa, the Levant, and the Gulf regions. Our data comprises 40 tasks and 14,575 multiple-choice questions in Modern Standard Arabic (MSA), and is carefully constructed by collaborating with native speakers in the region. 
<p align="left"> <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-circle.png?raw=true" style="width: 45%;" id="title-icon">       </p>

## Data
Each question in the dataset is a multiple-choice question with up to 5 choices and only one choice as the correct answer. 
The dataset can be accessed in [data](data) folder, and [Hugging Face](https://huggingface.co/datasets/MBZUAI/ArabicMMLU).

```
import datasets
data = datasets.load_dataset('MBZUAI/ArabicMMLU')
```

## Statistics

The data construction process involved a total of 10 Arabic native speakers from different countries: 6 internal workers (1 Jordanian, 1 Egyptian, 1 Lebanese, 1 from UAE, and 2 from KSA) and 4 external workers (3 Jordanian and 1 Egyptian).
The resulting corpus is sourced across the seven countries from which questions were collected, with Jordan, Egypt, and Palestine being the top three sources.
We categorize the collected questions into different subject areas, including: (1) STEM (Science, Technology, Engineering, and Mathematics); (2) Social Science; (3) Humanities; (4) Arabic Language; and (5) Others. 

<p align="left"> <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-country.png?raw=true" style="width: 45%;" id="title-icon">       </p>

## Examples

These questions are written in Arabic.

<p align="left"> 
    <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-ex2.png?raw=true" style="width: 45%;" id="title-icon"> 
    <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-ex1.png?raw=true" style="width: 45%;" id="title-icon">
</p>

## Evaluation

We evaluate 22 open-source multilingual models, 11 open-source Arabic-centric models, and 2 closed-source models. We experimented with different prompts in Arabic and English, and found the English prompt is the best. Below is the examples of input with the prompt.

<p align="left"> <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-prompt.png?raw=true" style="width: 30%;" id="title-icon">       </p>


#### Zero-shot Evaluation

 
<p align="left"> <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-result.png?raw=true" style="width: 70%;" id="title-icon">       </p>

#### Few-shot Evaluation

<p align="left"> 
    <img src="https://github.com/fajri91/eval_picts/blob/master/ArabicMMLU-fewshot.png?raw=true" style="width: 45%;" id="title-icon">
</p>

#### Evaluation
The code for the evaluation of each model we used is in `evaluate.py`, and the code to run them is listed in `run.sh`.

## Citation
```
@misc{koto2024arabicmmlu,
    title={"ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic"},
    author={"Fajri Koto and Haonan Li and Sara Shatanawi and Jad Doughman and Abdelrahman Boda Sadallah and Aisha Alraeesi and Khalid Almubarak and Zaid Alyafeai and Neha Sengupta and Shady Shehata and Nizar Habash and Preslav Nakov and Timothy Baldwin"},
    year={"2024"},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

The ArabicMMLU dataset is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
