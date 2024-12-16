# inaSpeechSegmenter
[![Python](https://img.shields.io/pypi/pyversions/inaSpeechSegmenter.svg?style=plastic)](https://badge.fury.io/py/inaSpeechSegmenter)
[![Python 3.7 to 3.12 unit tests](https://github.com/ina-foss/inaSpeechSegmenter/actions/workflows/python-package.yml/badge.svg)](https://github.com/ina-foss/inaSpeechSegmenter/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/inaSpeechSegmenter.svg)](https://badge.fury.io/py/inaSpeechSegmenter)
[![Docker Pulls](https://img.shields.io/docker/pulls/inafoss/inaspeechsegmenter)](https://hub.docker.com/r/inafoss/inaspeechsegmenter)

inaSpeechSegmenter is a CNN-based audio segmentation toolkit suited to the tasks of Voice Activity Detection and Speaker Gender Segmentation.


It splits audio signals into homogeneous zones of speech, music and noise.
Speech zones are split into segments tagged using speaker gender (male or female).
Male and female classification models are optimized for French language since they were trained using French speakers (acoustic correlates of speaker gender are language dependent).
Zones corresponding to speech over music or speech over noise are tagged as speech.
Singing voice is tagged as music.


## Highlights
* :trophy: ranked #1 against 6 open-source VAD system on a [French TV and radio benchmark](https://aclanthology.org/2024.lrec-main.785/)
* :transgender_flag: Extended as a non-binary Voice Gender Prediction system for [evaluating Transgender voice transition](https://doi.org/10.21437/Interspeech.2023-1835)
* :sunglasses: Used since 2020 in the annual [French Audivisual Regulation Authority report](https://www.arcom.fr/nos-ressources/etudes-et-donnees/mediatheque/la-representation-des-femmes-la-television-et-la-radio-rapport-sur-lexercice-2023) on women representation in TV and radio!
* :star2: Used to investigate the relationship between manual and automatic women representation descriptors [in French TV and radio](https://arxiv.org/abs/2406.10316)
* :female_sign: :male_sign: Applied for [large-scale gender representation studies](http://doi.org/10.18146/2213-0969.2018.jethc156) in French TV and radio.
* :partying_face: Won [MIREX 2018 speech detection challenge](http://www.music-ir.org/mirex/wiki/2018:Music_and_or_Speech_Detection_Results).


## Installation

inaSpeechSegmenter works with Python 3.7 to Python 3.12. It is based on Tensorflow which does not yet support Python 3.13+.

It is available on Python Package Index [inaSpeechSegmenter](https://pypi.org/project/inaSpeechSegmenter/) and packaged as a docker image [inafoss/inaspeechsegmenter](https://hub.docker.com/r/inafoss/inaspeechsegmenter).


### Prerequisites

inaSpeechSegmenter requires ffmpeg for decoding any type of format.
Installation of ffmpeg for ubuntu can be done using the following commandline:
```bash
$ sudo apt-get install ffmpeg
```

### PIP installation
```bash
# create a python 3 virtual environment and activate it
$ virtualenv -p python3 env
$ source env/bin/activate
# install framework and dependencies
$ pip install inaSpeechSegmenter
```

### Installing from from sources

```bash
# clone git repository
$ git clone https://github.com/ina-foss/inaSpeechSegmenter.git
# create a python 3 virtual environment and activate it
$ virtualenv -p python3 env
$ source env/bin/activate
# install framework and dependencies
# you should use pip instead of setup.py for installing from source
$ cd inaSpeechSegmenter
$ pip install .
# check program behavior
$ python setup.py test
```

## Using inaSpeechSegmenter

### Command-Line Interface
Binary program ina_speech_segmenter.py may be used to segment multimedia archives encoded in any format supported by ffmpeg. It requires input media and provide 2 segmentation output formats : csv (can be displayed with [Sonic Visualiser](https://www.sonicvisualiser.org)) and TextGrid ([Praat](https://www.fon.hum.uva.nl/praat/) format). Detailed command line options can be obtained using the following command :
```bash
# get help
$ ina_speech_segmenter.py --help
```

### Application Programming Interface

InaSpeechSegmentation API is intended to be very simple to use, and is illustrated by these 2 notebooks :
* [Google colab tutorial](https://colab.research.google.com/github/ina-foss/inaSpeechSegmenter/blob/master/tutorials/Demo_INASPeechSegmenter.ipynb): use API online
* [Jupyter notebook tutorial](tutorials/API_Tutorial.ipynb) : to be used offline

The class allowing to perform segmentations is called Segmenter.
It is the only class that you need to import in a program.
Class constructor accept 3 optional arguments:
* vad_engine (default: 'smn'). Allows to choose between 2 voice activity detection engines.
  * 'smn' is the more recent engine and splits signal into speech, music and noise segments
  * 'sm' was not trained with noise examples, and split signal into speech and music segments. Noise segments are either considered as speech or music. This engine was used in ICASSP study, and won MIREX 2018 speech detection challenge.
* detect_gender (default: True): if set to True, performs gender segmentation on speech segment and outputs labels 'female' or 'male'. Otherwise, outputs labels 'speech' (faster).
* ffmpeg: allows to provide a specific binary of ffmpeg instead of default system installation



## Citing

inaSpeechSegmenter has been presented at the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2018 conference in Calgary, Canada. If you use this toolbox in your research, you can cite the following work in your publications :


```bibtex
@inproceedings{ddoukhanicassp2018,
  author = {Doukhan, David and Carrive, Jean and Vallet, Félicien and Larcher, Anthony and Meignier, Sylvain},
  title = {An Open-Source Speaker Gender Detection Framework for Monitoring Gender Equality},
  year = {2018},
  organization={IEEE},
  booktitle={Acoustics Speech and Signal Processing (ICASSP), 2018 IEEE International Conference on}
}
```

inaSpeechSegmenter won [MIREX 2018 speech detection challenge](http://www.music-ir.org/mirex/wiki/2018:Music_and_or_Speech_Detection_Results)
Details on the speech detection submodule can be found below:

```bibtex
@inproceedings{ddoukhanmirex2018,
  author = {Doukhan, David and Lechapt, Eliott and Evrard, Marc and Carrive, Jean},
  title = {INA’S MIREX 2018 MUSIC AND SPEECH DETECTION SYSTEM},
  year = {2018},
  booktitle={Music Information Retrieval Evaluation eXchange (MIREX 2018)}
}
```

## Sibling Projects

* [inaFaceAnalyzer](https://github.com/ina-foss/inaFaceAnalyzer) : a Python toolbox for large-scale face-based description of gender representation in media with limited gender, racial and age biases
* [inaGVAD](https://github.com/ina-foss/InaGVAD) : a Challenging French TV and Radio Corpus annotated for Voice Activity Detection and Speaker Gender Segmentation


## CREDITS

This work has been partially funded by the French National Research Agency (project GEM : Gender Equality Monitor : ANR-19-CE38-0012) and by European Union's Horizon 2020 research and innovation programme (project [MeMAD](https://memad.eu) : H2020 grant agreement No 780069).

The code used to extract mel bands features is copy-pasted from [SIDEKIT project](https://git-lium.univ-lemans.fr/Larcher/sidekit)

Relevant contributions to the project were done by:
* [Eliott Lechapt](https://github.com/elechapt)
* [Cyril Lashkevich](https://github.com/notorca)
* [Rémi Uro](https://github.com/r-uro)
* [Simon Devauchelle](https://github.com/simonD3V)
* [Valentin Pelloin](https://github.com/valentinp72)
