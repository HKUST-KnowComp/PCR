# PCR
Codes for the survey paper:[A Brief Survey and Comparative Study of Recent Development of Pronoun Coreference Resolution](https://arxiv.org/abs/2009.12721) 
This reporsitory contains code and models adapted from [SpanBERT](https://github.com/younghz/Markdown), [Winogrande](https://github.com/allenai/winogrande), and [GPT2](https://github.com/openai/gpt-2). Additional, we include our analysis and replication code for the regular PCR (CoNLL-2012 solved by SpanBERT) and Hard PCR (Winograd Schema Challenge solved by Winogrande and GPT2) problems. The analysis includes finegrained pronoun setting, cross-domain setting, model comparison, dataset split and etc.


Please refer to the links of the repositories to learn about the setup of each model.

## Regular PCR
After the setup from the original repository, use `GPU=0 python evaluate.py <experiment>` for the evaluation. `independent.py` and `experiments.conf` have been modified to include our experments. 

## Hard PCR
### GPT-2
We have replicate the WSC experiment for GPT-2 in `./hard_PCR (WSC)/gpt2/src/gpt2_classification.ipynb`. The result is 69.23% accuracy in prediction.

### WinoGrande
You can run `./hard_PCR (WSC)/winogrande/scripts/run_experiment.py` with the command provided by the original repository. You can run `./hard_PCR (WSC)/winogrande/wsc_prune_exp.ipynb` to run the pruning experiement and  `./hard_PCR (WSC)/winogrande/data/finetuning_similarity_measurement.ipynb` for splitting the WSC alike datasets by the relevancy to the original 273 questions in WSC.

## Citation
Please check our arxiv draft [A Survey on Recent Progress of Pronoun Coreference Resolution](https://arxiv.org/abs/2009.12721) for more information.

## Contact
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](xzhaoar@connect.ust.hk), I will respond to that as soon as possible.
