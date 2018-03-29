# inaSpeechSegmenter

inaSpeechSegmenter is a framework for speech segmentation in Python 3.
It provides methods for speech music segmentation allowing to split audio signal into homogenous zones of speech and music.
It provides methods for speaker gender segmentation allowing to split speech excerpts into men and women speech.

## Installation
Python 3
tensorflow

```bash
$ virtualenv -p python3 inaSpeechSegEnv
$ source inaSpeechSegEnv/bin/activate
# install a backend for keras (tensorflow, theano, cntk...)
$ pip install tensorflow-gpu # for a GPU implementation
$ pip install tensorflow # for a CPU implementation
$ python setup.py install
```

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


In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), April 2018, Calgary, Canada




## CREDITS

This work was realized in the framework of MeMAD project.
https://memad.eu/
MeMAD is an EU funded H2020 research project.
It has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 780069.
