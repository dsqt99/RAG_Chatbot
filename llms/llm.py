from langchain_google_vertexai.chat_models import ChatVertexAI
from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool

class LLM:
    def __init__(self, modelname: str, search: bool = False):
        self.modelname = modelname

        if 'gemini' in self.modelname:
            if search:
                search_tools = [VertexTool(google_search={})]
                self.model = ChatVertexAI(
                    model=self.modelname,
                    temperature=0.5,
                    convert_system_message_to_human=True,
                    max_tokens=None
                ).bind_tools(search_tools)
                
            else:
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