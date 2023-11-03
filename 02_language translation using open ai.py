import os
from constants import openai_key
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = openai_key

language_prompt = PromptTemplate(
    input_variables = ["sentence", "language"],
    template = "Conver the {sentence} into language {language}"
)
input_sentence = "Hey, Good Morning Everyone. How are you?"
input_language = "Hindi"
language_prompt.format(sentence = input_sentence, language = input_language)

llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=language_prompt)
print(chain({'sentence': input_sentence, 'language': input_language}))


"""
Output:
{'sentence': 'Hey, Good Morning Everyone. How are you?', 'language': 'Hindi', 'text': '\n\nहेलो, सुप्रभात। आप कैसे हैं?'}
"""