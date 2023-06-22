
import streamlit as st
import pinecone
from bs4 import BeautifulSoup
from urllib.request import urlopen

def borrarJuego(bggID):
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], 
                environment=st.secrets["PINECONE_ENVIROMENT"])
    
    pineconeIndexName = st.secrets["PINECONE_INDEXNAME"]
    pineConePrefix = st.secrets["PINECONE_PREFIX"]
    pineconeIndex = pinecone.Index(index_name=pineconeIndexName)

    url = 'https://boardgamegeek.com/boardgame/'+str(bggID)
    soup = BeautifulSoup(urlopen(url), features="html5lib")
    BGGtitle = soup.title.get_text()
    guion = BGGtitle.find(' |')
    gameTitle = BGGtitle[0:guion]
    if not gameTitle == "BoardGameGeek":
        pineconeIndex.delete(delete_all=True, namespace=pineConePrefix+gameTitle)

borrarJuego(213606)


