# import fire
import random
import torch
import logging
# import argparse
from slam_llm.models.slam_model import slam_model
# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG
# from llama_recipes.configs import log_config as LOG_CONFIG

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
import os
import logging
from tqdm import tqdm
from slam_llm.models.slam_model import model_factory

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
	def to_plain_list(cfg_item):
		if isinstance(cfg_item, ListConfig):
			return OmegaConf.to_container(cfg_item, resolve=True)
		elif isinstance(cfg_item, DictConfig):
			return {k: to_plain_list(v) for k, v in cfg_item.items()}
		else:
			return cfg_item
	
	# kwargs = to_plain_list(cfg)
	kwargs = cfg
	log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
	
	logging.basicConfig(level=log_level)
	
	if kwargs.get("debug", False):
		import pdb;
		pdb.set_trace()
	
	main(kwargs)


def main(kwargs: DictConfig):

	# Update the configuration for the training and sharding process
	# train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
	# update_config((train_config, fsdp_config, model_config, log_config), **kwargs)
	train_config, fsdp_config, model_config, log_config, dataset_config = kwargs.train_config, \
	                                                                      kwargs.fsdp_config, \
	                                                                      kwargs.model_config, \
	                                                                      kwargs.log_config, \
	                                                                      kwargs.dataset_config

	OmegaConf.set_struct(kwargs,False)
	del kwargs["train_config"]
	del kwargs["fsdp_config"]
	del kwargs["model_config"]
	del kwargs["log_config"]
	del kwargs["dataset_config"]
	OmegaConf.set_struct(kwargs,True)

	# Set log
	if not os.path.exists(os.path.dirname(log_config.log_file)):
		os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
	logging.basicConfig(
		level=logging.INFO, 
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		filemode='w'
	)

	torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
	logger = logging.getLogger()  
	logger.setLevel(logging.INFO)

	file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
	file_handler.setLevel(logging.INFO)
	file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	file_handler.setFormatter(file_formatter)

	logger.handlers[0].setLevel(logging.INFO)
	console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	logger.handlers[0].setFormatter(console_formatter) 

	logger.addHandler(file_handler)
    
	logger.info("train_config: {}".format(train_config))
	logger.info("fsdp_config: {}".format(fsdp_config))
	logger.info("model_config: {}".format(model_config))

	
	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	# model_factory = get_custom_model_factory(model_config, logger)
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	model.to(torch.bfloat16)
	dataset_config["bf16"]=True
	model.to(device)
	model.eval()

	# dataset_config = generate_dataset_config(train_config, kwargs)
	logger.info("dataset_config: {}".format(dataset_config))
	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
	if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
		logger.info(f"--> Training Set Length = {len(dataset_test)}")

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			prefetch_factor=100,
			collate_fn=dataset_test.collator
        )
	

	logger.info("=====================================")
	pred_path = kwargs.get('decode_log') + "_pred"
	gt_path = kwargs.get('decode_log') + "_gt"
	preds = []
	gts = []
	
	import sacrebleu

	ave_bleu = 0.0
	all_bleu = 0.0
	count =1


	source = dataset_config.get("source", None)

	text_lan = source.split("_")[-2]
	if text_lan == "ja":
		text_lan = "ja-mecab"
	elif text_lan == "zh-CN":
		text_lan = "zh"
	else:
		text_lan = "13a"
	
	with open(pred_path, "w") as pred, open(gt_path, "w") as gt:
		for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
			for key in batch.keys():
				batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
			tmp_preds = []
			tmp_gts = []
			
			model_outputs = model.generate(**batch)
			output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
			for key, text, target in zip(batch["keys"], output_text, batch["targets"]):	
				print(key,text)
				print(key,target)
				pred.write(key + "\t" + text.replace("\n", " ") + "\n")
				gt.write(key + "\t" + target + "\n")



				tmp_preds.append(text)
				tmp_gts.append(target)


				preds.append(text)
				gts.append(target)
			
			tmp_result = sacrebleu.corpus_bleu(tmp_preds, [tmp_gts],tokenize=text_lan)
			print(tmp_result)

			
			all_bleu +=tmp_result.score
			ave_bleu = (all_bleu/count)
			count +=1
			print("ave_bleu:"+str(ave_bleu))

		result = sacrebleu.corpus_bleu(preds, [gts],tokenize=text_lan)
		print(result)



if __name__ == "__main__":
	main_hydra()