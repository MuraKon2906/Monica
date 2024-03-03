
import gradio as gr
import monica as md

sm = md.MonicaFunc()

def sentiment(How_are_you_feeling):
    result = sm.sentimentAn(How_are_you_feeling)
    return result

custom_css = """
body {
    background: linear-gradient(to right, red, white);
    display: flex;
    justify-content: center;
    align-items: center;
    height:   100vh;
    margin:   0;
}
"""

demo = gr.Interface(
    fn=sentiment,
    inputs=["text"],
    outputs=["text"],
    css=custom_css,
    #theme='ParityError/Anime',  # Use the Hugging Face theme for a cleaner look
    title="Sentiment analysis",
    description="Meet Monica, your digital emotions detective! 
    🌟 Simply input your thoughts, and Monica, powered by TensorFlow and Gradio, deciphers your feelings—happy or sad. 
    Express yourself, and let Monica unravel your emotions effortlessly. Try it now for a unique self-discovery journey! 
    🚀💬 #FeatureMonica"

)

demo.launch()
