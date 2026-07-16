"""
RAG系统的文档处理模块，实现功能：
1.定义文档加载器，根据不同的后缀名指定不同的文档加载器
2.文档加载：从指定文件夹加载多种类型文件并添加元数据
    1.遍历目录下的文件
    2.过滤文件类型，并加载文件
        2.1 根据文件类型来构造文档加载器。如果是 txt, csv,需要指定编码为 utf-8
        2.2 加载文件 并转为 Document
    3.给每个文档添加元数据：学科、路径、时间戳
3.文档分割：处理文档并进行两层切分，返回子块结果
    1.获取所有的Document
    2.初始化文档分割器
        2.1 对于markdown格式，使用MarkdownTextSplitter
        2.2 对于其他格式，使用自定义的 ChineseRecursiveTextSplitter
    3.把文件分割成多个子块
        3.1 把一个文档分割为多个父块chunk
        3.2 把每一个父块chunk 分割成多个子块sub_chunk
        3.3 给每个子块添加元数据，包括 parent_id, id, parent_content

面试题：

Q1: 文档加载和文档分块的作用分别是什么？
A1: 文档加载：将各种格式文件（PDF/Word/PPT/图片/Markdown等）统一转为LangChain的Document对象，便于后续处理。
    文档分块：将长文档切分为适合向量检索的小块（chunk），解决LLM上下文窗口限制，提高检索精度（大块语义模糊，小块过于碎片化）。

Q2: 为什么要进行父子分块（Parent-Child Chunking）？
A2: 两层分块策略兼顾检索精度和上下文完整性：
    - 父块（大chunk）：保留较完整的语义上下文，用于返回给LLM生成回答时提供充足背景信息。
    - 子块（小chunk）：粒度更细，用于向量化和检索匹配，提高检索的精确度。
    - 如果只用大chunk做检索，语义不精确；只用小chunk，返回内容过于碎片化。父子分块结合两者优点。
    - 每个子块通过 parent_id 关联父块，检索命中子块后可回溯获取父块完整内容。

Q3: 为什么要自定义文档加载模块，而不直接使用LangChain内置加载器？
A3: LangChain内置加载器不支持扫描版PDF、图片中文字、复杂Word/PPT的解析。教育场景中课件含大量截图、公式、表格，
    必须通过OCR提取图片文字，用pdfplumber/python-docx/python-pptx等库精确提取内容，保证知识完整性。

Q4: 为什么对图片做OCR？
A4: 教学课件（PPT/Word/PDF）中大量内容以图片形式存在（截图、公式、示意图），不做OCR会丢失这部分知识，
    导致RAG检索不到、回答不完整。OCR将图片中的文字转为可检索文本，确保知识库内容完整覆盖。

Q5: 如何提取PDF中的文本和图片？
A5: 使用pdfplumber逐页提取：
    - 文本：page.extract_text()直接提取纯文本。
    - 图片：page.images获取图片位置信息，通过pdfplumber的crop方法裁剪出图片区域，
      转为PIL Image对象后调用百度OCR API（basicAccurate）识别文字。
    - 每页的文本和图片OCR结果拼接为完整的页面内容，所有页面合并为一个Document返回。
    - 元数据保留总页数、当前页码、文档标题等信息。

Q6: Word、PPT、图片的加载原理是什么？
A6: - Word（.docx）：python-docx逐段提取文本；遍历表格提取单元格内容；
      通过document.part.related_parts提取嵌入图片，裁剪后OCR识别，所有结果拼接返回。
    - PPT（.pptx）：python-pptx获取每张幻灯片，通过slide.shapes提取文本和表格；
      使用pdf2image将幻灯片转为图片后OCR识别，所有结果拼接返回。
    - 图片（.jpg/.png）：PIL加载图片，调用百度OCR API的basicAccurate接口识别图片中的文字，
      将所有识别行拼接为完整文本返回。

Q7: 为什么要自定义文本分割模块（ChineseRecursiveTextSplitter），而不使用LangChain默认的RecursiveCharacterTextSplitter？
A7: LangChain默认分割器主要面向英文设计，中文标点（，。！？；：""）与英文标点（, . ! ? ; : ""）不同，
    默认分隔符无法正确切分中文文档。自定义分割器针对中文优化：
    - 支持中文段落分割（\n\n、\n）
    - 支持中文句子级分割（基于中文标点：。！？；等）
    - 参考LangChain RecursiveCharacterTextSplitter源码，扩展了中文正则分隔符和keep_separator=True参数。

Q8: chunk_size和chunk_overlap参数如何设置？各自的作用是什么？
A8: - chunk_size：每个文本块的最大字符数，影响检索粒度和上下文长度。
      值越大上下文越完整但检索精度下降，值越小检索越精确但语义可能不完整。
    - chunk_overlap：相邻文本块之间的重叠字符数，保证上下文连续性，避免语义在切分边界处被截断。
      通常设为chunk_size的10%~20%。
    - 父子分块中，父块chunk_size较大（如800-1000），子块较小（如200-400）。

"""

# core/document_processor.py
import os
# 文档加载器，把整个文档按照纯文本的形式加载成Document; 表格加载器，把表格数据转为Document对象
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredExcelLoader
# 文档加载器，把markdown格式的数据，提取文本内容，转成Document对象
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 支持markdown格式的切割（根据几级标题等）
from langchain_text_splitters import MarkdownTextSplitter

from datetime import datetime

# TODO 中文递归分割器（主要因为中文和英文的标点符号不一样，所以我们不使用langchain自带的RecursiveTextSplitter）
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter

# TODO PDF、DOC、PPT、IMG 格式的数据都是我们自己写的
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base.config import config
from base.logger import logger

# 1.定义文档加载器，根据不同的后缀名指定不同的文档加载器
DOCUMENT_LOADERS = {
    # txt 使用TextLoader
    ".txt": TextLoader,
    # PDF 使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader,
    # CSV 表格
    ".csv": CSVLoader,
    # Excel 表格（xlsx/xls）
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader
}

# 2.文档加载：从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path):
    """
    从指定文件夹加载多种类型文件并添加元数据, 处理文档并返回Document对象，用于后续进行文档分割
    :param directory_path: 要读取的路径，里面有RAG系统知识库的所有文档
    :return: 加载文档得到的Document对象的列表, list[Document]
    """
    # 定义一个空list,用于存放最终返回的结果
    documents = []

    # 获取所有支持的文件类型
    supported_extensions = DOCUMENT_LOADERS.keys()

    # 提取学科名称
    # D:\EduRAG_20260412\09-code_preview\00.project_code\integrated_qa_system\rag_qa\ai_data -> ai
    source = os.path.basename(directory_path).replace("_data","")

    # 1.遍历目录下的文件
    # 作用：递归遍历指定目录下面的所有文件、文件名
    # root: 当前所在的目录
    # dirs: 当前目录下有哪些文件夹
    # files: 当前目录下有哪些文件
    for root, _, files in os.walk(directory_path):
        for file in files:
            # 2.过滤文件类型，并加载文件
            # 获取当前文件的完整路径
            file_path = os.path.join(root, file)
            # 获取当前文件的后缀，比如.txt, .pdf, .docx
            extension_name = os.path.splitext(file)[-1].lower()

            # 判断当前文件格式，DOCUMENT_LOADERS是否支持
            if extension_name in supported_extensions:
                try:
                    # 2.1 根据文件类型来构造文档加载器。如果是 txt, csv,需要指定编码为 utf-8
                    loader_class = DOCUMENT_LOADERS[extension_name]
                    # 创建文档加载器对象,如果是 txt, csv,需要指定编码为 utf-8
                    if extension_name in [".txt", ".csv"]:
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 这样写等价于
                    # if extension_name == ".txt":
                    #     loader = TextLoader(file_path, encoding="utf-8")
                    # elif extension_name == ".pdf":
                    #     loader = OCRPDFLoader(file_path)
                    # ...

                    # 2.2 加载文件 并转为 Document
                    # 返回 加载好的完整文档对象, list[Document对象]
                    # 返回的 list中，只有一个元素（Document对象），这里只有一个文件
                    loaded_docs = loader.load()
                    # print(f"len(loaded_docs): {len(loaded_docs)}")
                    # print(f"loaded_docs: {loaded_docs}")

                    # 3.给每个文档添加元数据：学科、路径、时间戳
                    for doc in loaded_docs:
                        doc.metadata["source"] = source
                        doc.metadata["file_path"] = file_path
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                        doc.metadata["extension"] = extension_name
                        ...
                    # loaded_docs: [Document1,Document2]
                    # append: [[Document1,Document2],[Document3,Document4]]
                    # extend: [Document1,Document2,Document3,Document4]
                    documents.extend(loaded_docs)
                    logger.info(f"加载文件成功：{file_path}")

                except Exception as e:
                    logger.error(f"文件加载失败：{file_path}, error: {e}")
            else:
                logger.warning(f"不支持的文件类型：{file_path}")

    return documents

# 3.文档分割：处理文档并进行两层切分，返回子块结果
def process_documents(
    directory_path,
    parent_chunk_size=config.PARENT_CHUNK_SIZE,
    child_chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
):
    """
    处理文档并进行两层切分，返回子块结果
    :param directory_path:  要处理的文档所在文件夹路径
    :param parent_chunk_size:   父块大小
    :param child_chunk_size:    子块大小
    :param chunk_overlap:   重叠字符数
    :return:    所有子块列表
    """
    # 1.获取所有的Document
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载文档成功，共 {len(documents)} 个文档")

    # 2.初始化文档分割器
    # 2.1 对于markdown格式，使用MarkdownTextSplitter
    markdown_parent_splitter = MarkdownTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
    )
    markdown_child_splitter = MarkdownTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
    )

    # 2.2 对于其他格式，使用自定义的 ChineseRecursiveTextSplitter
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap,
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=chunk_overlap,
    )

    # 3.把文档分割成多个子块
    # 初始化空列表，用于存储所有子块
    child_chunks = []
    # 遍历原始文档，带上索引 i
    for i, doc in enumerate(documents):
        # 3.1 把一个文档分割为多个父块chunk
        # 获取文件扩展名，比如.md, .txt, .pdf, .docx
        # print(f"doc.metadata: {doc.metadata}")
        file_extension = doc.metadata.get("extension",".pdf").lower()
        # print(f"file_extension: {file_extension}")

        # 选择分割器
        is_markdown = file_extension in [".md", ".markdown"]
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f"正在处理文档：{doc.metadata.get('file_path', 'unknown')}, 分割器: {parent_splitter_to_use.__class__.__name__}")

        # 进行文档分割
        # 需要传入list格式, list[Document对象]
        parent_docs = parent_splitter_to_use.split_documents([doc])
        logger.info(f"父块数量: {len(parent_docs)}")
        # print(f"parent_docs: {parent_docs}")

        # 3.2 把每一个父块chunk 分割成多个子块sub_chunk
        #  遍历父块，带上索引 j
        for j, parent_doc in enumerate(parent_docs):
            # 生成父块ID: doc_{i}_parent_{j}
            parent_id = f"doc_{i}_parent_{j}"
            # 将 父块ID 添加到元数据
            parent_doc.metadata["parent_id"] = parent_id
            # 将父块 分割为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            # 3.3 给每个子块添加元数据，包括 parent_id, id, parent_content
            # 遍历子块，带上索引 k
            for k, sub_chunk in enumerate(sub_chunks):
                # 添加 父块ID 到子块元数据
                sub_chunk.metadata["parent_id"] = parent_id
                # 添加 父块内容 到子块元数据
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # 添加 子块ID 到子块元数据,子块ID: {parent_id}_child_{k}
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                # 添加子块 到 子块列表
                child_chunks.append(sub_chunk)
    # 记录子块数量
    logger.info(f"子块总数量: {len(child_chunks)}")
    return child_chunks

# 主程序
if __name__ == '__main__':
    # 1.文档加载
    # documents = load_documents_from_directory(directory_path=r"D:\EduRAG_V7.5_gz7\07-live_code\01.project_code_gz7"
    #                                              r"\integrated_qa_system\rag_qa\ai_data")
    # 2.文档分块
    child_chunks = process_documents(
        os.path.join(config.PROJECT_ROOT,r"D:\EduRAG_V7.5_gz7\07-live_code\01.project_code_gz7\integrated_qa_system\rag_qa\ai_data")
    )
    print(child_chunks)
