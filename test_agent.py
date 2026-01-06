from langchain.llms import OpenAI 
llm = OpenAI(openai_api_key="test", model_name="text-davinci-003", temperature=0.1) 
print("? Configuration API ancienne OK") 
