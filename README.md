# PCR
Codes for the survey paper on pronoun coreference resolution
This reporsitory contains code and models adapted from [SpanBERT](https://github.com/younghz/Markdown), [Winogrande](https://github.com/allenai/winogrande), and [GPT2](https://github.com/openai/gpt-2). Additional, we include our analysis and replication code for the regular PCR (CoNLL-2012 solved by SpanBERT) and Hard PCR (Winograd Schema Challenge solved by Winogrande and GPT2) problems. The analysis includes finegrained pronoun setting, cross-domain setting, model comparison, dataset split and etc.


Please refer to the links of the repositories to learn about the setup of each model.

## Regular PCR
After the setup from the original repository, use `GPU=0 python evaluate.py <experiment>` for the evaluation. `independent.py` and `experiments.conf` have been modified to include our experments. 

## Hard PCR
TODO

## Citation
Please check our arxiv draft [A Survey on Recent Progress of Pronoun Coreference Resolution](https://www.google.com/search?q=404&oq=404&aqs=chrome..69i57.1935j0j7&sourceid=chrome&ie=UTF-8) for more information.

## Contact
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](xzhaoar@connect.ust.hk), I will respond to that as soon as possible.
