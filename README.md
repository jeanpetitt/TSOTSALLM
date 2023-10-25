# TSOTSALLM: Large Language Models at large scales
Large Language Models (LLMs) have made a new waive in Artificial Intelligence. Used by software giants, they are regarded as key enablers in various domains such as education, medicine, etc. Given the computational resources and the training time, their use is currently unattainable for the majority of companies and private users. In fact, their development and maintenance is currently only possible for large organizations able to afford the corresponding costs.

In developing countries particularly where LLMs can be used to foster the development as well as attaining the Sustainable Development Goals, this technology is almost inaccessible to researchers, students and start-ups. In these countries, many researchers use their computers to build AI models. For instance, in a recent work, we combined several computers for annotating tabular datasets using Knowledge Graphs. Even if this work got accepted at SemTab@ISWC, we were not able to run our model on the overall datasets and we did not get good evaluations.

TSOTSALLM aims to scale LLMS, making them accessible to researchers, students and a wide audience of companies of all sizes, particularly to people from developing countries. Thus, this work aims to provide methodologies and scale LLMS to be accessible to a wide audience of users globally.

## Research methodology
The research methodology relies on empirical research methods in software engineering. It consists of combining action research with case study research. The action research allows us to explore, test, evaluate different approaches for reducing LLMs size so as to determine the best one. The case study research aims to test on one case and generalize to the other cases. In this work, we started with the case Llama 2.

### Research objective
The objective of this work is to explore, test and evaluate different approaches to reducing Llama 2 so as to make it accessible to a large audience.

### Approaches to test
We identified different ways to fine-tune LLMs so as to reduce it size: technique:

#### Approach 1
This approach consists to use PEFT (Parameter-Efficient Fine-Tuning), LoRA (Low-Rank Adaptation) and QLoRA (Quantized Low-Rank Adaptation) techniques that allow us to train Llama 2 with drastic reduction of RAM. This may allow us to fine tune Llama 2 on a single GPU of reasonable size.

* The usual step to train Llama 2 consists firstly, an intensive pre-training on billions or trillions of token to obtain a base model and then we can start the fine-tuning process on this model to specialize it on specific tasks.

* **PEFT** is used to add new layers (adapters) on top of the pre-trained base model and reduce the storage requirements. This technique allowed us to enhance the reusability and portability of the model.

* **LoRA** is used to avoid the problem of latence during the inference step. To this end, values are added to the parameters of the LLM. Thus, using LoRA, we are able to train and store the changes of additional weights of the pre-trained model.

* **QLoRA** is used to reduce the memory footprint and the optimization of the GPU. To this end, the LoRA method is quantized.

#### Approach 2
The second approach we are planning to set up is a neuro-symbolic approach. It consists of using a short description of the data in the form of a graph to define the set of parameters that should be selected in the LLM. Thus, using this graph, the appropriate parameters to be used to train the LLM is selected.

This approach is still under development.

Once we experiment the two approaches, we are planning to set up additional experimentations and ablation studies given the results obtained.

### Methodology description
The methodology is an ongoing process of reflection and revision, during which appropriate solutions to the problem have to be found. It consists of:
* **Planning:** this consists of identifying a way to reduce LLMs and plan its development
* **Implementation:** this consists of implementing the methods identified
* **Revise:** this consists of testing and revising the approach that is just implemented
* **Re-implement:** based on the revised, the approach can be re-implemented.

### Experimentation

#### Experimentation environment
The current experimentation environment consists of the free <strong>Google Colab</strong> having an NVIDIA T4 GPU (16GB) available for a period less than 2h30 min. This environment can be used to train the  Llama 2 on only <strong>10% of the dataset</strong>. When more data is added, the RAM fills up and the processing cannot continue.

Using this development environment, we are currently working on the first approach presented previously and a first result was submitted to the boot.


### Contacts

* jeanpetityvelos@gmail.com
* fidel.jiomekong@facsciences-uy1.cm
