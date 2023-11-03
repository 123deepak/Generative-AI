import os
from constants import openai_key
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = openai_key

examples = [
    {"number": "81", "output": "9"},
    {"number": "121", "output": "11"}
]

example_formatter_template = """number: {number} | output: {output}"""

example_prompt = PromptTemplate(
    input_variables = ["number", "output"],
    template=example_formatter_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt=example_prompt,
    suffix="number: {input} | output:",
    input_variables=["input"]
)

input = 225
print(few_shot_prompt.format(input=input))
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=few_shot_prompt)
print(chain({'input': input}))


"""
python Few_Shot_Learnings_OpenAI.py

Output:
number: 81 | output: 9  
number: 121 | output: 11
number: 225 | output:   
{'input': 225, 'text': ' 15'}
"""