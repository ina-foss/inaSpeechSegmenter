# inaSpeechSegmenter
[![Python](https://img.shields.io/pypi/pyversions/inaSpeechSegmenter.svg?style=plastic)](https://badge.fury.io/py/inaSpeechSegmenter)
[![Python 3.6 to 3.10 unit tests](https://github.com/ina-foss/inaSpeechSegmenter/actions/workflows/python-package.yml/badge.svg)](https://github.com/ina-foss/inaSpeechSegmenter/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/inaSpeechSegmenter.svg)](https://badge.fury.io/py/inaSpeechSegmenter)
[![Docker Pulls](https://img.shields.io/docker/pulls/inafoss/inaspeechsegmenter)](https://hub.docker.com/r/inafoss/inaspeechsegmenter)

inaSpeechSegmenter is a CNN-based audio segmentation toolkit.


It splits audio signals into homogeneous zones of speech, music and noise.
Speech zones are split into segments tagged using speaker gender (male or female).
Male and female classification models are optimized for French language since they were trained using French speakers (accoustic correlates of speaker gender are language dependent).
Zones corresponding to speech over music or speech over noise are tagged as speech.


inaSpeechSegmenter has been designed in order to perform [large-scale gender equality studies](http://doi.org/10.18146/2213-0969.2018.jethc156) based on men and women speech-time percentage estimation.

## Installation

inaSpeechSegmenter works with Python 3.6+.
It is available on Python Package Index [inaSpeechSegmenter](https://pypi.org/project/inaSpeechSegmenter/) and packaged as a docker image [inafoss/inaspeechsegmenter](https://hub.docker.com/r/inafoss/inaspeechsegmenter).


### Prerequisites

inaSpeechSegmenter requires ffmpeg for decoding any type of format.
Installation of ffmpeg for ubuntu can be done using the following commandline:
```bash
$ sudo apt-get install ffmpeg
```

### PIP installation
```bash
# create a python 3 virtual environement and activate it
$ virtualenv -p python3 env
$ source env/bin/activate
# install framework and dependencies
$ pip install inaSpeechSegmenter
```

### Installing from from sources

```bash
# clone git repository
$ git clone https://github.com/ina-foss/inaSpeechSegmenter.git
# create a python 3 virtual environement and activate it
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

### Speech Segmentation Program
Binary program ina_speech_segmenter.py may be used to segment multimedia archives encoded in any format supported by ffmpeg. It requires input media and output csv files corresponding to the segmentation. Corresponding csv may be visualised using softwares such as https://www.sonicvisualiser.org/
```bash
# get help
$ ina_speech_segmenter.py --help
usage: ina_speech_segmenter.py [-h] -i INPUT [INPUT ...] -o OUTPUT_DIRECTORY [-d {sm,smn}] [-g {true,false}] [-b FFMPEG_BINARY] [-e {csv,textgrid}]

Do Speech/Music(/Noise) and Male/Female segmentation and store segmentations into CSV files. Segments labelled 'noEnergy' are discarded from music, noise, speech and gender
analysis. 'speech', 'male' and 'female' labels include speech over music and speech over noise. 'music' and 'noise' labels are pure segments that are not supposed to contain speech.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input media to analyse. May be a full path to a media (/home/david/test.mp3), a list of full paths (/home/david/test.mp3 /tmp/mymedia.avi), a regex input
                        pattern ("/home/david/myaudiobooks/*.mp3"), an url with http protocol (http://url_of_the_file)
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Directory used to store segmentations. Resulting segmentations have same base name as the corresponding input media, with csv extension. Ex: mymedia.MPG will
                        result in mymedia.csv
  -d {sm,smn}, --vad_engine {sm,smn}
                        Voice activity detection (VAD) engine to be used (default: 'smn'). 'smn' split signal into 'speech', 'music' and 'noise' (better). 'sm' split signal into
                        'speech' and 'music' and do not take noise into account, which is either classified as music or speech. Results presented in ICASSP were obtained using 'sm'
                        option
  -g {true,false}, --detect_gender {true,false}
                        (default: 'true'). If set to 'true', segments detected as speech will be splitted into 'male' and 'female' segments. If set to 'false', segments
                        corresponding to speech will be labelled as 'speech' (faster)
  -b FFMPEG_BINARY, --ffmpeg_binary FFMPEG_BINARY
                        Your custom binary of ffmpeg
  -e {csv,textgrid}, --export_format {csv,textgrid}
                        (default: 'csv'). If set to 'csv', result will be exported in csv. If set to 'textgrid', results will be exported to praat Textgrid

Detailled description of this framework is presented in the following study: Doukhan, D., Carrive, J., Vallet, F., Larcher, A., & Meignier, S. (2018, April). An open-source speaker
gender detection framework for monitoring gender equality. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 5214-5218). IEEE.
```
### Using Speech Segmentation API

InaSpeechSegmentation API is intended to be very simple to use.
The class allowing to perform segmentations is called Segmenter.
It is the only class that you need to import in a program.
Class constructor accept 3 optional arguments:
* vad_engine (default: 'smn'). Allows to choose between 2 voice activity detection engines.
  * 'smn' is the more recent engine and splits signal into speech, music and noise segments
  * 'sm' was not trained with noise examples, and split signal into speech and music segments. Noise segments are either considered as speech or music. This engine was used in ICASSP study, and won MIREX 2018 speech detection challenge.
* detect_gender (default: True): if set to True, performs gender segmentation on speech segment and outputs labels 'female' or 'male'. Otherwise, outputs labels 'speech' (faster).
* ffmpeg: allows to provide a specific binary of ffmpeg instead of default system installation


See the following notebook for a comprehensive example: [API Tutorial Here!](API_Tutorial.ipynb)

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

inaSpeechSegmenter won MIREX 2018 speech detection challenge.  
http://www.music-ir.org/mirex/wiki/2018:Music_and_or_Speech_Detection_Results  
Details on the speech detection submodule can be found bellow:  


```bibtex
@inproceedings{ddoukhanmirex2018,
  author = {Doukhan, David and Lechapt, Eliott and Evrard, Marc and Carrive, Jean},
  title = {INA’S MIREX 2018 MUSIC AND SPEECH DETECTION SYSTEM},
  year = {2018},
  booktitle={Music Information Retrieval Evaluation eXchange (MIREX 2018)}
}
```


## CREDITS

This work has been partially funded by the French National Research Agency (project GEM : Gender Equality Monitor : ANR-19-CE38-0012) and by European Union's Horizon 2020 research and innovation programme (project [MeMAD](https://memad.eu) : H2020 grant agreement No 780069).

Some optimization within inaSpeechSegmenter code were realized by Cyril Lashkevich
https://github.com/notorca

The code used to extract mel bands features is copy-pasted from sidekit project:
https://git-lium.univ-lemans.fr/Larcher/sidekit

Relevant contributions to the project were done by:
* Eliott Lechapt : https://github.com/elechapt
* Rémi Uro : https://github.com/r-uro
