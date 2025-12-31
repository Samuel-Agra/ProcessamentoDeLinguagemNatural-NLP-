import pandas as pd
import numpy as np
import re
import nltk
from math import log2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Verificar as mensagens de tweets, detectar: spam, bots, trending... ou crar features: clustering, classificação,
# classificação... comparar entropia e sentimento


#Calcula a aleatorideda dos bits, "quantidade diferente de digitos" - Entropia de Shannon
def calc_informacao(texto):
    #verificar se o texto existe ou se é uma str
    if not texto or not isinstance(texto, str):
        return 0
    # pre-processamento (tokenização)
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", texto)
    texto = re.sub(r"[^a-záàâãéèêíïóôõöúçñ\s]", "", texto)
    tokens = word_tokenize(texto, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    tokens_limpos = [t for t in tokens if t not in stop_words and len(t) > 2]
    texto_limpo = " ".join(tokens_limpos)
    return texto_limpo

    #transformação para probabilidade
    frequencias = {c: texto.count(c) / len(texto) for c in set(texto)}
    #calculo da entropia
    return -sum(p * log2(p) for p in frequencias.values())

# Calcula a entropia (informação) de cada tweet e
# a informação média por palavra, e adiciona esses valores ao DataFrame.
def analisar_tweets(caminho_dataset):
    # lê o caminho dos tweets e atribui a df
    df = pd.read_csv(caminho_dataset)

    # verifica se os dados do dataset estão completos
    if 'tweet_text' not in df.columns:
        raise ValueError("O dataset precisa ter a coluna 'tweet_text'.")
    if 'sentiment' not in df.columns:
        raise ValueError("O dataset precisa ter a coluna 'sentiment'.")

    info_por_tweet = []
    info_por_palavra = []

    #
    for tweet in df['tweet_text']:
        if isinstance(tweet, str):
            info_total = calc_informacao(tweet)
            palavras = tweet.split()
            media_palavra = info_total / len(palavras) if palavras else 0 #calcula a entropia por palavra
            info_por_tweet.append(info_total)
            info_por_palavra.append(media_palavra)
        else: #caso o tweet seja vazio ou não str
            info_por_tweet.append(0)
            info_por_palavra.append(0)

    df['info_tweet'] = info_por_tweet #entropia do tweet (texto)
    df['info_palavra'] = info_por_palavra #entropia das palavras

    # Média geral
    media_geral = {
        "media_info_tweet": np.mean(df['info_tweet']),  # entropia geral dos tweets
        "media_info_palavra": np.mean(df['info_palavra'])  # media de palavras nos tweets
    }
     #imprime os resutados
    print("\nMédia geral de informação:")
    for k, v in media_geral.items():
        print(f"  {k}: {v:.4f}")


    #  "df.groupby('sentiment')" -> separa os tweets por classe de sentimentos (negativo, neutro e positivo)
    #  "[['info_tweet', 'info_palavra']]" -> seleciona apenas as colunas relevanes
    #  ".mean()" -> calcula a média
    medias_sentimento = df.groupby('sentiment')[['info_tweet', 'info_palavra']].mean()
    print("\nMédia de informação por sentimento:")
    print(medias_sentimento)

    #"salva no dataframe"
    df.to_csv("resultado_informacao_tweets.csv", index=False)
    print("\nResultados salvos em: resultado_informacao_tweets.csv")

# --- Execução ---
if __name__ == "__main__":
    caminho = "NoThemeTweets.csv"
    analisar_tweets(caminho)


### Após o cálculo da quantidade de informação de cada tweet, foram obtidas métricas agregadas,
### incluindo a média global de informação por tweet e por palavra. Em seguida, os dados foram agrupados
### por classe de sentimento, permitindo comparar a complexidade informacional do texto entre sentimentos distintos.
### Por fim, os resultados foram exportados para um arquivo CSV para análise posterior.
