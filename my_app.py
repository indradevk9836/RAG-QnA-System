import streamlit as st
from QnA_Sys import get_prompt, load_llm, get_vectorstore
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("🧠 Question Answering System")

# PDF Upload Section
st.header("📄 Upload Your Documents")
uploaded_files = st.file_uploader(
    "Choose PDF files", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload one or more PDF files to create your knowledge base"
)

# Process uploaded files
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Create a unique key for caching based on file names and sizes
    file_info = [(f.name, f.size) for f in uploaded_files]
    
    # Load and process PDFs with caching
    with st.spinner("Processing your documents..."):
        try:
            # Use cached vectorstore creation
            vectorstore = get_vectorstore(uploaded_files)
            st.session_state.vectorstore = vectorstore
            st.session_state.docs_loaded = True
            st.session_state.file_info = file_info
            st.success("📚 Documents processed and ready for questions!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.session_state.docs_loaded = False
else:
    st.info("👆 Please upload PDF files to get started")
    st.session_state.docs_loaded = False

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Chat input
prompt = st.chat_input("Ask questions about your uploaded documents...")

if prompt and prompt.strip():
    # Check if documents are loaded
    if not st.session_state.get('docs_loaded', False):
        st.error("⚠️ Please upload PDF documents first!")
        st.stop()
    
    # Add user message to chat
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
    Don't provide anything outside the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    try:
        # Get vectorstore from session state
        vectorstore = st.session_state.vectorstore
        
        if vectorstore is None:
            st.error("Failed to load the vector store")
            st.stop()

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': get_prompt(prompt_template, input_variables=["context", "question"])}
        )

        # Get response
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
        
        # Display response
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role': 'assistant', 'content': result})

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Reset documents button
        if st.button("🔄 Reset Documents"):
            if 'vectorstore' in st.session_state:
                del st.session_state.vectorstore
                st.session_state.docs_loaded = False
                st.session_state.messages = []
                st.rerun()

    except Exception as e:
        import traceback
        st.error(f"Error: {str(e)}")
        st.text(traceback.format_exc())

