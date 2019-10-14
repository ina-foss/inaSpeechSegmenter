# inaSpeechSegmenter

inaSpeechSegmenter is a CNN-based audio segmentation toolkit.


It splits audio signals into homogeneous zones of music and speech.
Speech zones are split into segments tagged using speaker gender (male or female).
Male and female classification models are optimized for French language since they were trained using French speakers (accoustic correlates of speaker gender are language dependent).
Zones corresponding to speech over music are tagged as speech.


inaSpeechSegmenter has been designed in order to perform large-scale gender equality studies based on men and women speech-time percentage estimation.

## Installation

inaSpeechSegmenter is a framework in python 3.
It can be installed using the following procedure:

### Prerequisites

inaSpeechSegmenter requires ffmpeg for decoding any type of format.
Installation of ffmpeg for ubuntu can be done using the following commandline:
```bash
$ sudo apt-get install ffmpeg
```
### PIP installation
Simplest installation procedure
```bash
# create a python 3 virtual environement and activate it
$ virtualenv -p python3 inaSpeechSegEnv
$ source inaSpeechSegEnv/bin/activate
# install a backend for keras (tensorflow, theano, cntk...)
$ pip install tensorflow-gpu # if you wish GPU implementation (recommended if your host has a GPU)
$ pip install tensorflow # for a CPU implementation
# install framework and dependencies
$ pip install inaSpeechSegmenter
```

### Installing from from sources

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
# check program behavior
$ python setup.py check
```

## Using inaSpeechSegmenter

### Speech Segmentation Program
Binary program ina_speech_segmenter.py may be used to segment multimedia archives encoded in any format supported by ffmpeg. It requires input media and output csv files corresponding to the segmentation. Corresponding csv may be visualised using softwares such as https://www.sonicvisualiser.org/
```bash
# get help
$ ina_speech_segmenter.py --help
usage: ina_speech_segmenter.py [-h] -i INPUT [INPUT ...] -o OUTPUT_DIRECTORY

Does Speech/Music and Male/Female segmentation. Stores segmentations into CSV
files

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input media to analyse. May be a full path to a media
                        (/home/david/test.mp3), a list of full paths
                        (/home/david/test.mp3 /tmp/mymedia.avi), or a regex
                        input pattern ("/home/david/myaudiobooks/*.mp3")
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Directory used to store segmentations. Resulting
                        segmentations have same base name as the corresponding
                        input media, with csv extension. Ex: mymedia.MPG will
                        result in mymedia.csv
```
### Using Speech Segmentation API

InaSpeechSegmentation API is intended to be very simple to use.
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

This work was realized in the framework of MeMAD project.
https://memad.eu/
MeMAD is an EU funded H2020 research project.
It has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 780069.
