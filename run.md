# 1. Platform
	Please install PyTorch by yourself. (https://pytorch.org/get-started/locally/).
# 2. Codes
	The codes of baseline models (BLOCK and LXMERT) are included in the folds  './block/block_bl' and './lxmert/lxmert_bl', respectively. 
	And the proposed strategy with both models are included in the folds './block/block_cr' and './lxmert/lxmert_cr', respectively. 
# 3. Data
	Please download and process all the needed data first (include image data, question data, pretrained model data and word embedding data), you can do this by following the methods in these two works.
   	 ## 1) Code of BLOCK: https://github.com/Cadene/block.bootstrap.pytorch.
    	 ## 2) Code of LXMERT: https://github.com/airsplay/lxmert.
# 4. Path
	Please replace all the 'root_pth' in the above codes as your own real data path.
# 5. Train.
	For codes in folders './block/block_bl'  and './block/block_cr' , training the model by run 'CUDA_VISIBLE_DEVICES="0" python main.py  --cp_name block'.
	For codes in folders './lxmert/lxmert_bl' and './lxmert/lxmert_cr', training the model by run 'bash run/vqa_ft.bash 0 vqa_lxmert'.
# 6. Evaluation
	All the evalution results will be printed on the screen after each train epoch.
# 7. Other Problem
	If you have other questions about this work or code, please feel easy to contact us.


PS：Special thanks to the authors of VQAv2, BLOCK and LXMERT, the datasets and the codes used in this research project.