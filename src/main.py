from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from get_mongoDB import MongoDBClient
import os
from dotenv import load_dotenv
import logging

# Set up page config
st.set_page_config(
    page_title="Trợ Lý Lý Thuyết Đồ Thị",
    page_icon="📊",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Google API
google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)

# Create MongoDB connection
@st.cache_resource
def initialize_mongo():
    return MongoDBClient()._connect()

mongo_client = initialize_mongo()
db = mongo_client.get_database('chatbotdb')

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model='models/gemini-2.0-flash-thinking-exp-01-21',
        temperature=0.5,
        max_tokens=None,
        timeout=60,
        max_retries=2,
        convert_system_message_to_human=True,
        api_key=google_api_key
    )

llm = initialize_llm()

# Create prompt template
revised_template = ChatPromptTemplate.from_messages([
    ('system', """**Vai trò:** Bạn là Trợ lý AI chuyên sâu về Lý thuyết Đồ thị, một cộng tác viên am hiểu và nhiệt tình.

**Nhiệm vụ chính:** Giải quyết các câu hỏi và bài toán về lý thuyết đồ thị mà người dùng đưa ra. Điều này bao gồm giải thích lý thuyết, giải bài tập và **viết mã code** cho các thuật toán liên quan khi người dùng yêu cầu.

**Sử dụng Thông tin Tham khảo (`context`):**
1.  **Hiểu rõ Câu hỏi:** Luôn bắt đầu bằng việc phân tích kỹ yêu cầu **cụ thể** của người dùng.
2.  **Phân tích `context`:** Xác định và phân biệt thông tin từ nguồn 'knowledge' (lý thuyết) và 'exercises' (bài tập mẫu).
3.  **Tổng hợp Thông minh:** Kết hợp lý thuyết và ví dụ một cách linh hoạt. Dùng lý thuyết để soi sáng phương pháp trong bài tập mẫu, và dùng bài tập mẫu để minh họa cho lý thuyết. **Tuyệt đối không sao chép máy móc.**
4.  **Áp dụng Thực tế:** Vận dụng kiến thức đã tổng hợp vào **đúng số liệu và điều kiện** trong câu hỏi của người dùng.
5.  **Xử lý Thiếu thông tin:** Nếu `context` không đủ hoặc không liên quan để trả lời, hãy **nêu rõ điều này**. Cố gắng trả lời dựa trên kiến thức chung nếu có thể, hoặc **đặt câu hỏi làm rõ** cho người dùng.

**Cách Trả lời:**
* **Ưu tiên sự Rõ ràng:** Trình bày mọi giải thích, lời giải theo các bước logic, mạch lạc, dễ hiểu.
* **Giải thích Lý thuyết:** Định nghĩa rõ ràng, nêu tên định lý (nếu phổ biến), giải thích cặn kẽ các khái niệm.
* **Giải Bài tập:** Trình bày chi tiết quá trình giải, giải thích từng bước tính toán. **Làm nổi bật đáp số cuối cùng.**
* **Viết Mã Code (Nếu yêu cầu hoặc phù hợp):**
    * Cung cấp các đoạn **mã code rõ ràng, có thể chạy được** (ưu tiên Python).
    * Giải thích logic của code, các biến quan trọng, và cách sử dụng/thực thi.
    * Nêu rõ các giả định hoặc giới hạn (ví dụ: độ phức tạp, kiểu dữ liệu đầu vào).
* **Chủ động & Hỗ trợ:** Nếu thấy phù hợp, có thể đề cập ngắn gọn đến các khái niệm liên quan hoặc phương pháp thay thế. Đừng ngại **hỏi lại** nếu yêu cầu của người dùng chưa rõ ràng.
"""),
    ('human', "Thông tin tham khảo:\n```\n{context}\n```\n\nCâu hỏi của tôi:\n{question}")
])

# Function to get embeddings
def get_embedding(text, model='models/gemini-embedding-exp-03-07'):
    result = genai.embed_content(model=model, content=text)
    return result["embedding"]

# Function to create context
def create_context_refined(
    query,
    db,
    knowledge_coll="knowledge",
    exercises_coll="exercises",
    vector_index="vector_index",
    embedding_field="embedding",
    knowledge_fields_map={"title": "head", "content": "define"},
    exercises_fields_map={"problem": "problem.content", "solution": "solution.content"},
    results_per_coll=5,
    num_candidates_multiplier=10,
    total_limit=10
):
    context_str = "Không tìm thấy thông tin liên quan."
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            logger.error("Lỗi: Không thể tạo embedding cho query.")
            return "Lỗi khi xử lý query embedding."

        num_candidates = results_per_coll * num_candidates_multiplier

        def build_projection(collection_name, fields_map):
            projection = {
                "_id": 0,
                "source": {"$literal": collection_name},
                "score": {"$meta": "vectorSearchScore"}
            }
            for display_name, actual_field in fields_map.items():
                projection[display_name] = f"${actual_field}"
            return projection

        knowledge_projection = build_projection(knowledge_coll, knowledge_fields_map)
        exercises_projection = build_projection(exercises_coll, exercises_fields_map)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": vector_index,
                    "path": embedding_field,
                    "queryVector": query_embedding,
                    "limit": results_per_coll,
                    "numCandidates": num_candidates
                }
            },
            { "$project": knowledge_projection },
            {
                "$unionWith": {
                    "coll": exercises_coll,
                    "pipeline": [
                        {
                            "$vectorSearch": {
                                "index": "exercises",
                                "path": embedding_field,
                                "queryVector": query_embedding,
                                "limit": results_per_coll,
                                "numCandidates": num_candidates
                            }
                        },
                        { "$project": exercises_projection }
                    ]
                }
            },
            { "$sort": { "score": -1 } },
            { "$limit": total_limit }
        ]

        results = list(db[knowledge_coll].aggregate(pipeline))

        if results:
            formatted_context = ["Thông tin tham khảo được tìm thấy:"]
            for i, doc in enumerate(results):
                source = doc.get('source', 'N/A')
                score = doc.get('score', 0)
                item_str = f"\n--- (Kết quả {i+1}) ---\n"

                fields_map = knowledge_fields_map if source == knowledge_coll else exercises_fields_map
                for display_name in fields_map.keys():
                     item_str += f"{display_name.capitalize()}: {doc.get(display_name, 'N/A')}\n"

                formatted_context.append(item_str.strip())

            context_str = "\n".join(formatted_context)

        return context_str

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi tạo context: {e}", exc_info=True)
        return f"Đã xảy ra lỗi khi truy xuất thông tin: {e}"

# Format documents
def format_docs(docs):
    return "\n\n".join([doc.get('define', '') for doc in docs if isinstance(doc, dict) and 'define' in doc])

# Create chain
@st.cache_resource
def create_chain():
    return (
        RunnableMap({
            "context": lambda x: x["context"],
            "question": lambda x: x["question"]
        })
        | revised_template
        | llm
        | StrOutputParser()
    )

chain = create_chain()

# Streamlit UI
st.title("🧠 Trợ Lý Lý Thuyết Đồ Thị")
st.markdown("""
Chào mừng bạn đến với Trợ lý AI chuyên về Lý thuyết Đồ Thị.
Hãy đặt câu hỏi về các khái niệm, định lý, hoặc bài tập đồ thị, tôi sẽ cố gắng giúp bạn!
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
query = st.chat_input("Hãy nhập câu hỏi về lý thuyết đồ thị...")

# Process query
if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant thinking
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Đang tìm kiếm thông tin...")
        
        # Get context
        with st.spinner("Đang tìm kiếm thông tin liên quan..."):
            context = create_context_refined(query, db)
        
        # Process with LLM
        with st.spinner("Đang phân tích và soạn câu trả lời..."):
            result = chain.invoke({"context": context, "question": query})
        
        # Display result
        message_placeholder.markdown(result)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})

# Sidebar with information
with st.sidebar:
    st.header("Về ứng dụng này")
    st.markdown("""
    Ứng dụng này sử dụng công nghệ RAG (Retrieval Augmented Generation) để:
    
    1. Tìm kiếm thông tin liên quan từ cơ sở dữ liệu MongoDB
    2. Phân tích và tổng hợp thông tin
    3. Tạo câu trả lời chi tiết cho người dùng
    
    Cơ sở dữ liệu chứa lý thuyết và bài tập về Lý thuyết Đồ thị.
    """)
    
    
    
    