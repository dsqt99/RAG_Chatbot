from langchain_google_vertexai.chat_models import ChatVertexAI

class LLM:
    def __init__(self, modelname: str):
        self.modelname = modelname

        if 'gemini' in self.modelname:
            self.model = ChatVertexAI(
                model=self.modelname, 
                temperature=0.5,
                convert_system_message_to_human=True,
                max_tokens=None
            )

        else:
            raise Exception(f'Modelname {self.modelname} is not available, we only support gemini model.')
        
    def invoke(self, text):
        return self.model.invoke(text)#.content