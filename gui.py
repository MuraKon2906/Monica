
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
    description="feature of Monica Project :)"

)

demo.launch()
