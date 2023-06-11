
import pinecone
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from urllib.request import urlopen

def borrarJuego(bggID):
    config = dotenv_values()
    pinecone.init(api_key=config["PINECONE_API_KEY"], 
                environment=config["PINECONE_ENVIROMENT"])
    
    pineconeIndexName = config["PINECONE_INDEXNAME"]
    pineConePrefix = config["PINECONE_PREFIX"]
    pineconeIndex = pinecone.Index(index_name=pineconeIndexName)

    url = 'https://boardgamegeek.com/boardgame/'+str(bggID)
    soup = BeautifulSoup(urlopen(url), features="html5lib")
    BGGtitle = soup.title.get_text()
    guion = BGGtitle.find(' |')
    gameTitle = BGGtitle[0:guion]
    if not gameTitle == "BoardGameGeek":
        pineconeIndex.delete(delete_all=True, namespace=pineConePrefix+gameTitle)

borrarJuego(239959)


