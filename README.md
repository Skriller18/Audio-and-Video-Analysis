# Audio and Video Analysis

This is an application that will do facial recognition for the people present in the video as well as transcibe video along with performing Speaker Diarization for the given audio.

## Steps to be followed:

Clone this Git Repository
```bash
git clone https://github.com/Skriller18/Audio-and-Video-Analysis.git
```

Install the needed requirements
```bash
pip install -r requirements.txt
```

Load the images if you have into the faces folder by keeping the name of the image as the name of the label
```bash
cd cast
```

Edit the Labels file for the name of the label you want to ID for each photo
```bash
gedit labels.txt
```

Run the encoding script to get the encoding for each person
```bash
python encodings.py
```

Select the audio model that you want to use for transcribing and Speaker Diarization. The models used for this case are whisper and pyannote/speaker-diarization.

Run the main script using the Streamlit Interface which will load a web-hosted site on the localhost can be acccesed by using the port 8501
```bash
streamlit run main.py
Local URL: http://localhost:8501
```

## Python version preferred : 3.10.14