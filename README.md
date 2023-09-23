# TSOTSALLM: Large Language Models at large scales
Large Language Models (LLMs) have made a new waive in Artificial Intelligence. Used by software giants, they are regarded as key enablers in various domains such as education, medicine, etc. Given the computational resources and the training time, their use is currently unattainable for the majority of companies and private users. In fact, their development and maintenance is currently only possible for large organizations able to afford the corresponding costs. 

In developing countries particularly where LLMs can be used to foster the development as well as attaining the Sustainable Development Goals, this technology is almost inaccessible to researchers, students and start-ups. In these countries, many researchers use their computers to build AI models. For instance, in a recent work, we combined several computers for annotating tabular datasets using Knowledge Graphs \cite{}. Even if this work got accepted at SemTab@ISWC, we were not able to run our model on the overall datasets and we did not get good evaluations.

TSOTSALLM aims to scale LLMS, making them accessible to researchers, students and a wide audience of companies of all sizes, particularly to people from developing countries. Thus, this work aims to provide methodologies and scale LLMS to be accessible to a wide audience of users globally.

## Research methodology
The research methodology relies on empirical research method in software engineering. It consists of combining action research with case study research. The action research allow to explore, test, evaluate different approaches for reducing LLMs size so as to determine the best one. The case study research aims to test on one case and generalize to the other cases. In this work, we started with the case Llama 2.

### Research objective
The objective of this work is to explore, test and evaluate different approaches of reducing Llama 2 so as to make it acccessible to a large audience.

### Approaches to test
We identified different ways to fine-tune LLMs so as to reduce it size: technique 1, 2, ..., n

#### Technique 1
Picture + description
this approach consist to use PEFT, LoRA and QLoRA techniques that allow us to train our LLM with drastic reduction of RAM requirements and consequently  allowing to fine tune this models on a single GPU of reaseonable size.

The usual step to train an LLM consist firstly, an intensive pre-training on billions or trillions of token to obtain a base model and then we can start the fine-tuning process on this model to specialize it on a specific tasks. it is in this phase that the PEFT has its purpose.

* Peft is used to add the news parameters or layer on top from pre-trained base model. that layers is generally called <b>Adapters</b> and the techniques for fitting them is called <strong> Adaptation Tuning </strong>. We use this technique to reduce RAM and storage requirement by only fine. 
Furthermoree, it enhances the reutilisablity and portability of the model. as the small control point obtained can be easily added to the base model and this model can be easily refine and reused  in multiple scenarios by adding the peft parametters.

* In Low-Rank Adaption for LLM, the idea is not to include the new layers but to add values to the parameters in a way to avoid the problem of latence in inference step. using LoRA allows us to train and store the changes of the addtional weights of the pre-trained model. 
* In QloRA we quantize the LoRa method allowing 4-bit normal quantization, nf4, a type optimized for normally distributed weights,double quantization to reduce the memory footprint and the optimization of the GPU unified memory like NVIDIA





#### Technique 2
Picture + description


### Methodology description
The methodology is an ongoing process of reflexion and revision, during which appropriate solutions to the problem have to be found. It consists of:
* **Planning:** this consists of identifying a way to reduce LLMs and plan its development
* **Implementation:** this consists of implementing the methods identified
* **Revise:** this consists of testing and revising the approach that is just implemented
* **Re-implement:** based on the revised, the approach can be re-implemented



### Experimentation
 we used 10% of our dataset for fine-tune LLama 2.
We did this to be able to experiment with our approaches, as we didn't have sufficient computational resources.
since we don't have any local resources available to run our LLM, we used the development environment <strong>Google Colab</strong>, which has an NVIDIA T4 GPU (16GB) available for a period of around 2h30 min. which doesn't allow us to run the LLM on our entire dataset.