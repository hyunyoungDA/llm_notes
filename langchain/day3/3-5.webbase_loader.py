# 웹 URL에서 HTML을 불러와서 텍스트로 변환 가능

import bs4
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

# 뉴스기사 내용을 로드
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    
    # IP 차단 우회하기 위해 프록시 설정 
    proxies = {
      "http": "http://{username}:{password}:@proxy.service.com:6666/",
      "https":"https://{username}:{password}:@proxy.service.com:6666/"
    },
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            # 속성 설정 -> css selector 
            # WebPage에 따라 selector 구조 잘 확인한 후 결정
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    # User-Agent 설정해서 크롤러가 아닌 사용자가 이용하는 것처럼.
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

docs = loader.load()
print(f"문서의 수: {len(docs)}")
# print(docs)

# 여러 웹페이지르 한 번에 로드 
# Web_paths를 리스트로  
loader2 = WebBaseLoader(
    web_paths=[
        "https://n.news.naver.com/article/437/0000378416",
        "https://n.news.naver.com/mnews/hotissue/article/092/0002340014?type=series&cid=2000063",
    ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div", # <div> 태그만 선택 
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

# 데이터 로드
docs = loader.load()

# 문서 수 확인
print(len(docs))