# Inspiration from https://huggingface.co/spaces/vumichien/whisper-speaker-diarization

import whisper
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from pytube import YouTube
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

from zipfile import ZipFile
from io import StringIO
import csv

# ---- Model Loading ----

whisper_models = ["base", "small", "medium", "large"]
source_languages = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
}

source_language_list = [key[0] for key in source_languages.items()]

MODEL_NAME = "openai/whisper-small"
lang = "en"

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ---- S2T & Speaker diarization ----

def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    text = pipe(file)["text"]

    return warn_output + text


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def convert_to_wav(filepath):
    _,file_ending = os.path.splitext(f'{filepath}')
    audio_file = filepath.replace(file_ending, ".wav")
    print("starting conversion to wav")
    os.system(f'ffmpeg -i "{filepath}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
    return audio_file


def speech_to_text(microphone, file_upload, selected_source_lang, whisper_model, num_speakers):
    """
    # Transcribe audio file and separate into segment, assign speakers to segments
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    model = whisper.load_model(whisper_model)
    time_start = time.time()

    try:
        # Read and convert audio file
        warn_output = ""
        if (microphone is not None) and (file_upload is not None):
            warn_output = (
                "WARNING: You've uploaded an audio file and used the microphone. "
                "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
            )

        elif (microphone is None) and (file_upload is None):
            return "ERROR: You have to either use the microphone or upload an audio file"

        file = microphone if microphone is not None else file_upload
        
        if microphone is None and file_upload is not None:
            file = convert_to_wav(file)
        
        # Get duration
        with contextlib.closing(wave.open(file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=3, best_of=3)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(file, **transcribe_options)
        segments = result["segments"]
        print("whisper done with transcription")
    except Exception as e:
        raise RuntimeError("Error converting audio file")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        if num_speakers == 1:
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER 1'
        else:
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        if num_speakers == 1:
            objects['Start'].append(str(convert_time(segment["start"])))
            objects['Speaker'].append(segment["speaker"])
            for (i, segment) in enumerate(segments):
                text += segment["text"] + ' '
            objects['Text'].append(text)
            objects['End'].append(str(convert_time(segment["end"])))
        else:
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    objects['Start'].append(str(convert_time(segment["start"])))
                    objects['Speaker'].append(segment["speaker"])
                    if i != 0:
                        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                        objects['Text'].append(text)
                        text = ''
                text += segment["text"] + ' '
            objects['End'].append(str(convert_time(segments[i - 1]["end"])))
            objects['Text'].append(text)
        
        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
        system_info = f"""
        *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
        *Processing time: {time_diff:.5} seconds.*
        *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
        """

        return pd.DataFrame(objects), system_info
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)

# ---- Youtube Conversion ----

def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("Success download video")
    print(abs_video_path)
    return abs_video_path



def yt_to_text(video_file_path, selected_source_lang, whisper_model, num_speakers):
    """
    # Transcribe youtube link using OpenAI Whisper
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    model = whisper.load_model(whisper_model)
    time_start = time.time()
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)

    try:
        # Read and convert youtube video
        _,file_ending = os.path.splitext(f'{video_file_path}')
        print(f'file ending is {file_ending}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
        
        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = model.transcribe(audio_file, **transcribe_options)
        segments = result["segments"]
        print("starting whisper done with whisper")
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        if num_speakers == 1:
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER 1'
        else:
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        if num_speakers == 1:
            objects['Start'].append(str(convert_time(segment["start"])))
            objects['Speaker'].append(segment["speaker"])
            for (i, segment) in enumerate(segments):
                text += segment["text"] + ' '
            objects['Text'].append(text)
            objects['End'].append(str(convert_time(segment["end"])))
        else:
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    objects['Start'].append(str(convert_time(segment["start"])))
                    objects['Speaker'].append(segment["speaker"])
                    if i != 0:
                        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                        objects['Text'].append(text)
                        text = ''
                text += segment["text"] + ' '
            objects['End'].append(str(convert_time(segments[i - 1]["end"])))
            objects['Text'].append(text)
        
        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
        system_info = f"""
        *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
        *Processing time: {time_diff:.5} seconds.*
        *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
        """

        return pd.DataFrame(objects), system_info
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)

def download_csv(dataframe: pd.DataFrame):
    compression_options = dict(method='zip', archive_name='output.csv')
    dataframe.to_csv('output.zip', index=False, compression=compression_options)  
    return 'output.zip'

# ---- Gradio Layout ----
# Inspiration from https://huggingface.co/spaces/vumichien/whisper-speaker-diarization

# -- General Functions --
df_init = pd.DataFrame(columns=['Start', 'End', 'Speaker', 'Text'])
memory = psutil.virtual_memory()
title = "Whisper speaker diarization & speech recognition"
interface = gr.Blocks(title=title)
interface.encrypt = False

# -- Functions Audio Input --
microphone_in = gr.inputs.Audio(source="microphone", 
                                type="filepath", 
                                optional=True)

upload_in = gr.inputs.Audio(source="upload", 
                            type="filepath", 
                            optional=True)

selected_source_lang_audio = gr.Dropdown(choices=source_language_list, 
                                         type="value", 
                                         value="en", 
                                         label="Spoken language in audio", 
                                         interactive=True)

selected_whisper_model_audio = gr.Dropdown(choices=whisper_models, 
                                           type="value", 
                                           value="base", 
                                           label="Selected Whisper model", 
                                           interactive=True)

number_speakers_audio = gr.Number(precision=0, 
                                  value=2, 
                                  label="Selected number of speakers", 
                                  interactive=True)

system_info_audio = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")

transcription_df_audio = gr.DataFrame(value=df_init,
                                      label="Transcription dataframe", 
                                      row_count=(0, "dynamic"), 
                                      max_rows = 10, 
                                      wrap=True, 
                                      overflow_row_behaviour='paginate')

csv_download_audio = gr.outputs.File(label="Download CSV")

# -- Functions Video Input --
video_in = gr.Video(label="Video file", 
                    mirror_webcam=False)

youtube_url_in = gr.Textbox(label="Youtube url", 
                            lines=1, 
                            interactive=True)

selected_source_lang_yt = gr.Dropdown(choices=source_language_list, 
                                      type="value", 
                                      value="en", 
                                      label="Spoken language in audio", 
                                      interactive=True)

selected_whisper_model_yt = gr.Dropdown(choices=whisper_models, 
                                        type="value", 
                                        value="base", 
                                        label="Selected Whisper model", 
                                        interactive=True)

number_speakers_yt = gr.Number(precision=0, 
                               value=2, 
                               label="Selected number of speakers", 
                               interactive=True)

system_info_yt = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")

transcription_df_yt = gr.DataFrame(value=df_init,
                                   label="Transcription dataframe", 
                                   row_count=(0, "dynamic"), 
                                   max_rows = 10, 
                                   wrap=True, 
                                   overflow_row_behaviour='paginate')

csv_download_yt = gr.outputs.File(label="Download CSV")

with interface:
    with gr.Tab("Whisper speaker diarization & speech recognition"):
        gr.Markdown('''
            <div>
            <h1 style='text-align: center'>Whisper speaker diarization & speech recognition</h1>
            This space uses Whisper models from <a href='https://github.com/openai/whisper' target='_blank'><b>OpenAI</b></a> to recoginze the speech and ECAPA-TDNN model from <a href='https://github.com/speechbrain/speechbrain' target='_blank'><b>SpeechBrain</b></a> to encode and clasify speakers</h2>
            </div>
        ''')

        with gr.Row():
            gr.Markdown('''
            ### Transcribe youtube link using OpenAI Whisper
            ##### 1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
            ##### 2. Generating speaker embeddings for each segments.
            ##### 3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
            ''')             

        with gr.Row():
            with gr.Column():
                microphone_in.render()
                upload_in.render()
                with gr.Column():
                    gr.Markdown('''
                    ##### Here you can start the transcription process.
                    ##### Please select the source language for transcription.
                    ##### You should select a number of speakers for getting better results.
                    ''')
                selected_source_lang_audio.render()
                selected_whisper_model_audio.render()
                number_speakers_audio.render()
                transcribe_btn = gr.Button("Transcribe audio and initiate diarization")
                transcribe_btn.click(speech_to_text, 
                                    [
                                        microphone_in,
                                        upload_in,
                                        selected_source_lang_audio,
                                        selected_whisper_model_audio,
                                        number_speakers_audio
                                    ],
                                    [
                                        transcription_df_audio,
                                        system_info_audio
                                    ])

                
        with gr.Row():
            gr.Markdown('''
            ##### Here you will get transcription  output
            ##### ''')
            

        with gr.Row():
            with gr.Column():
                transcription_df_audio.render()
                system_info_audio.render()
        
        with gr.Row():
            with gr.Column():
                download_btn = gr.Button("Download transcription dataframe")
                download_btn.click(download_csv, transcription_df_audio, csv_download_audio)
                csv_download_audio.render()
            
        with gr.Row():
            gr.Markdown('''Chair of Data Science and Natural Language Processing - University of St. Gallen''')
    
    with gr.Tab("Youtube Speech to Text"):
        with gr.Row():
            gr.Markdown('''
                        <div>
                        <h1 style='text-align: center'>Youtube Speech Recognition & Speaker Diarization</h1>
                        </div>
                        ''')
            
        with gr.Row():
            gr.Markdown('''
                        ### Transcribe Youtube link
                        #### Test with the following examples:
                        ''')
            examples = gr.Examples(examples = 
                                   [
                                       "https://www.youtube.com/watch?v=vnc-Q8V4ihQ",
                                       "https://www.youtube.com/watch?v=_B60aTHCE5E",
                                       "https://www.youtube.com/watch?v=4BdKZxD-ziA",
                                       "https://www.youtube.com/watch?v=4ezBjAW26Js",
                                   ],
                                   label="Examples UNISG",
                                   inputs=[youtube_url_in])
        
        with gr.Row():
            with gr.Column():
                youtube_url_in.render()
                download_youtube_btn = gr.Button("Download Youtube video")
                download_youtube_btn.click(get_youtube, [youtube_url_in], [video_in])
                print(video_in)
        
        with gr.Row():
            with gr.Column():
                video_in.render()
                with gr.Column():
                    gr.Markdown('''
                                #### Start the transcription process.
                                #### To initiate, please select the source language for transcription.
                                #### For better performance select the number of speakers.
                                ''')
                selected_source_lang_yt.render()
                selected_whisper_model_yt.render()
                number_speakers_yt.render()
                transcribe_btn = gr.Button("Transcribe audio and initiate diarization")
                transcribe_btn.click(yt_to_text, 
                                    [
                                        video_in,
                                        selected_source_lang_yt,
                                        selected_whisper_model_yt,
                                        number_speakers_yt
                                    ],
                                    [
                                        transcription_df_yt,
                                        system_info_yt
                                    ])
        
        with gr.Row():
            gr.Markdown('''
                        #### Here you will get transcription  output
                        #### ''')
        
        with gr.Row():
            with gr.Column():
                transcription_df_yt.render()
                system_info_yt.render()

        with gr.Row():
            with gr.Column():
                download_btn = gr.Button("Download transcription dataframe")
                download_btn.click(download_csv, transcription_df_audio, csv_download_yt)
                csv_download_yt.render()
        
        with gr.Row():
            gr.Markdown('''Chair of Data Science and Natural Language Processing - University of St. Gallen''')


def main():
    interface.launch(share=True, server_name="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
