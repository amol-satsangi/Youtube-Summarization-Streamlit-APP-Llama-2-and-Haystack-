# YouTube Video URL
# Pytube library: For video manipulation
# Whisper model: For transription
# Prompt: Summarization Prompt
# LLM: Llama 2 GGUF (32 K context Size)

# How to load Llama 2 in Haystack?
# Using custom llama.cpp class

# Summarization: Text Output


import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptModel, PromptNode
from haystack.nodes.audio import WhisperTranscriber 
from haystack.pipelines import Pipeline 
import time 
from model_add import LlamaCPPInvocationLayer

# To make the sidebar wider for better viewing of code and output
st.set_page_config(
    layout = "wide",
    page_title = "Youtube Summary")

def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

def initialize_model(full_path):
    return PromptModel(
        model_name_path = full_path,
        invocation_layer_class = LlamaCPPInvocationLayer,
        use_gpu = False,
        max_length = 512
    )    

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path = model, default_prompt = summary_prompt, use_gpu = False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component = whisper, name = "Whisper Transcription", inputs = ["File"])
    pipeline.add_node(component = prompt_node, name = "prompt", input = ["whisper"])
    output = Pipeline.run(file_path = [file_path])
    return output

def main():
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI Anytime.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")


    if st.button("Submit") and youtube_url:
        file_path = download_video(youtube_url)
        full_path = "Youtube Summarization/llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = initialize_prompt_node(model)
        #         print (transcribe_audio([file_path], prompt_node))
        output = transcribe_audio(file_path, prompt_node)

        col1, col2 = st.columns([1,1])
        
        with col1:
            st.video(youtube_url)
            #             st.image(output["File"][0]["data"]["content"], caption="Summarized Audio")

        with col2:
            st.header("Summarization of the youtube Video")
            st.write(output)
            st.success(output["resutls"][0].split("\n\n[INST]")[0])

if __name__ == "__main__":
    main()
        