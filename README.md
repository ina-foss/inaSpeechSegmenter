# inaSpeechSegmenter

inaSpeechSegmenter is a CNN-based audio segmentation toolkit.


It splits audio signals into homogeneous zones of speech, music and noise.
Speech zones are split into segments tagged using speaker gender (male or female).
Male and female classification models are optimized for French language since they were trained using French speakers (accoustic correlates of speaker gender are language dependent).
Zones corresponding to speech over music or speech over noise are tagged as speech.


inaSpeechSegmenter has been designed in order to perform [large-scale gender equality studies](http://doi.org/10.18146/2213-0969.2018.jethc156) based on men and women speech-time percentage estimation.

## Installation

inaSpeechSegmenter is a framework in python 3. Only python versions greater or equal to 3.6 are supported.
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
$ python setup.py test
```

## Using inaSpeechSegmenter

### Speech Segmentation Program
Binary program ina_speech_segmenter.py may be used to segment multimedia archives encoded in any format supported by ffmpeg. It requires input media and output csv files corresponding to the segmentation. Corresponding csv may be visualised using softwares such as https://www.sonicvisualiser.org/
```bash
# get help
$ ina_speech_segmenter.py --help
usage: ina_speech_segmenter.py [-h] -i INPUT [INPUT ...] -o OUTPUT_DIRECTORY
                               [-d {sm,smn}] [-g {true,false}]

Do Speech/Music(/Noise) and Male/Female segmentation and store segmentations
into CSV files. Segments labelled 'noEnergy' are discarded from music, noise,
speech and gender analysis. 'speech', 'male' and 'female' labels include
speech over music and speech over noise. 'music' and 'noise' labels are pure
segments that are not supposed to contain speech.

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
  -d {sm,smn}, --vad_engine {sm,smn}
                        Voice activity detection (VAD) engine to be used
                        (default: 'smn'). 'smn' split signal into 'speech',
                        'music' and 'noise' (better). 'sm' split signal into
                        'speech' and 'music' and do not take noise into
                        account, which is either classified as music or
                        speech. Results presented in ICASSP were obtained
                        using 'sm' option
  -g {true,false}, --detect_gender {true,false}
                        (default: 'true'). If set to 'true', segments detected
                        as speech will be splitted into 'male' and 'female'
                        segments. If set to 'false', segments corresponding to
                        speech will be labelled as 'speech' (faster)
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

This work was realized in the framework of MeMAD project.
https://memad.eu/
MeMAD is an EU funded H2020 research project.
It has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 780069.

Some optimization within inaSpeechSegmenter code were realized by Cyril Lashkevich
https://github.com/notorca
