### SESR: Simultaneous Enhancement and Super-Resolution 
- Perceptually enhanced image generation at higher spatial scales
- Paper: https://arxiv.org/pdf/2002.01155.pdf

![im2](/data/p1.jpg)

![im2](/data/p2.jpg)

![im3](/data/p3.jpg)


### Deep SESR model
- An efficient model for underwater imagery; can be trained end-to-end for 2x-4x SESR 
- Model architecture and implementation details: https://arxiv.org/pdf/2002.01155.pdf
- Weights for Deep SESR 2x with 1D FENet (trained on UFO-120) are provided in models/
	- HDF5: deep_sesr_2x_1d.h5 file; use [test_sesr_Keras.py](test_sesr_Keras.py)
	- Protobuf: deep_sesr_2x_1d.pb file; use [test_sesr_TF.py](test_sesr_TF.py) 


### UFO-120 dataset
- 1500 training and 120 test samples (underwater images) 
- Facilitates paired training of 2x, 3x, and 4x SESR models 
- Also has annotated saliency maps for training saliency prediction models 
- Can be downloaded from: http://irvlab.cs.umn.edu/resources/ufo-120-dataset


#### Bibliography entry for citation:
	```
	@article{islam2020sesr,
	    title={Simultaneous Enhancement and Super-Resolution of Underwater Imagery 
                   for Improved Visual Perception},
	    author={Islam, Md Jahidul and Luo, Peigen and Sattar, Junaed},
	    journal={To appear in Robotics: Science and Systems (RSS). arXiv:2002.01155},
	    year={2020}
	}
	```

#### More details: soon...
