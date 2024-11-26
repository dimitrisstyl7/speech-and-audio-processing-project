# Speech and Audio Processing: Word Detection System

## [University of Piraeus](https://www.unipi.gr/en/home/) | [Department of Informatics](https://cs.unipi.gr/en/)
**BSc course**: Speech and Audio Processing

**Semester**: 8

**Project Completion Year**: 2024

## Description
This project implements a word detection system that segments spoken sentences into individual words using various classifiers.
The system is designed to analyze audio recordings of speech and return the time intervals of the detected words. It aims to provide 
an efficient and speaker-independent solution for speech recognition tasks, utilizing classifiers such as Least Squares, SVM, RNN, and MLP.

## How to Run
1. **Clone the repository**:
```bash
git clone https://github.com/dimitrisstyl7/speech-and-audio-processing-project-2024.git
```
2. **Navigate to the project directory**:
```bash
cd speech-and-audio-processing-project-2024
```
3. **Create and activate a virtual environment**:

_On Linux/Mac_
```bash
python3 -m venv venv
source venv/bin/activate
```

_On Windows_
```bash
python -m venv venv
venv\Scripts\activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```
5. **Run the program - Classifiers Training**:

_On Linux/Mac_
```bash
python3 train_classifiers.py
```
_On Windows_
```bash
python train_classifiers.py
```
6. **Run the program - Word Detection**:

_On Linux/Mac_
```bash
python3 word_detector.py
```
_On Windows_
```bash
python word_detector.py
```

## Notes
- To successfully play audio through the `word_detector.py` program, you must have [VLC media player](https://www.videolan.org/vlc/) installed on your computer.

## Acknowledgments
This project was developed as part of the "Speech and Audio Processing" BSc course at the University of Piraeus. Contributions and feedback are always welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
