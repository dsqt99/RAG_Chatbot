import sys
sys.path.append("../")

import os
from dotenv import load_dotenv
from llms.llm import LLM
from pydantic import BaseModel
from google.oauth2 import service_account
import vertexai
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

load_dotenv(override=True)

credentials = service_account.Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"), credentials=credentials)
embedding = VertexAIEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
llm = LLM(modelname=os.getenv("GEMINI_PRO_MODEL")).model

try:
    vectorstore = FAISS.load_local(
        os.getenv("FAISS_PATH"),
        embedding,
        allow_dangerous_deserialization=True
    )   
except:
    vectorstore = None

system_template = """Bạn là một trợ lý ảo chuyên nghiệp của công ty Công ty TNHH mua bán nợ Việt Nam DATC, được đào tạo đặc biệt để trả lời các câu hỏi liên quan đến các quy định, nghị định, chính sách và quy trình nội bộ của công ty.
Nhiệm vụ chính của bạn:
- Dựa trên thông tin truy xuất (retrieval) dưới đây, hãy tạo ra câu trả lời rõ ràng, súc tích và chính xác bằng tiếng Việt.
- **Ưu tiên trình bày thông tin theo đúng thứ tự xuất hiện của các đoạn văn bản gốc được truy xuất.** 
- Yêu cầu BẮT BUỘC về Trích dẫn ID: Đối với mỗi phần thông tin cụ thể bạn trình bày trong câu trả lời, bạn PHẢI trích dẫn rõ ràng mã định danh (ID) của văn bản/đoạn văn bản gốc mà thông tin đó được lấy từ đó.
- Độ chính xác và Trung lập: Luôn đảm bảo câu trả lời là chính xác dựa trên tài liệu gốc. Không suy diễn, không đưa ra ý kiến cá nhân, lời khuyên pháp lý hoặc thông tin không có trong cơ sở dữ liệu. 
- Giữ giọng văn chuyên nghiệp, lịch sự và hữu ích.
Lưu ý:
- Định dạng trích dẫn ID: Sử dụng định dạng ([[mã_id_văn_bản]]) ngay sau thông tin liên quan hoặc ở cuối câu/đoạn văn tóm tắt thông tin từ nguồn đó.
- Ví dụ:
    - "Công ty DATC có tổng công ty là 1 tỷ đồng. [[id_1]]"
    - "Công ty DATC có tổng công ty là 1 tỷ đồng. [[id_1]][[id_2]]"
    - "Công ty DATC có tổng công ty là 1 tỷ đồng. [[id_1]][[id_2]][[id_3]]"
"""