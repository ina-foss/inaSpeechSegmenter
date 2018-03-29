# inaSpeechSegmenter

inaSpeechSegmenter is a framework for speech segmentation in Python 3.
It provides methods for speech music segmentation allowing to split audio signal into homogenous zones of speech and music.
It provides methods for speaker gender segmentation allowing to split speech excerpts into men and women speech.

## Installation

inaSpeechSegmenter is a framework in python 3. It can be installed using the following procedure:


```bash
# clone git repository
$ git clone https://github.com/ina-foss/inaSpeechSegmenter.git
# create a python 3 virtual environement and activate it
$ virtualenv -p python3 inaSpeechSegEnv
$ source inaSpeechSegEnv/bin/activate
# install a backend for keras (tensorflow, theano, cntk...)
$ pip install tensorflow-gpu # if you wish GPU implementation (recommended)
$ pip install tensorflow # for a CPU implementation
# install framework and dependencies
$ cd inaSpeechSegmenter
$ python setup.py install
```

## Speech Segmentation Program

## Using Speech Segmentation API

InaSpeechSegmentation API is very simple to use!
See the following notebook for a comprehensive example: [API Tutorial Here!](API_Tutorial.ipynb)

## Citing

inaSpeechSegmenter has been presented at the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2018 conference in Calgary, Canada. If you use this toolbox in your research, you can cite the following work in your publications :


```bibtex
@inproceedings{ddoukhanicassp2018,
  author = {Doukhan, David and Carrive, Jean and Vallet, Félicien and Larcher, Anthony and Meignier, Sylvain},
  title = {An Open-Source Speaker Gender Detection Framework for Monitoring Gender Equality},
  year = {2018},
  organization={IEEE},
  booktitle={Acoustics Speech and Signal Processing (ICASSP), 2010 IEEE International Conference on}
}
```


## CREDITS

This work was realized in the framework of MeMAD project.
https://memad.eu/
MeMAD is an EU funded H2020 research project.
It has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 780069.
